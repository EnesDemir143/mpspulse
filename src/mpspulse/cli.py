"""mpspulse CLI (argparse, torch-free until a real measurement runs)."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys

EXIT_USAGE = 2

EXPLAIN = """\
GPU event time vs wall-clock time
=================================
gpu_event_ms: hardware-measured time the GPU spent executing kernels
  (MPS event pairs, one sync at the end). Excludes Python overhead,
  dataloader waits, and CPU-side optimizer work.
wall_ms: end-to-end time per iteration (data + CPU + GPU + optimizer),
  measured with torch.mps.synchronize() + time.perf_counter().
data_ms: time spent fetching the next batch between iterations.
  If data_ms/wall_ms > 0.2, data loading is the bottleneck.
samples_per_sec: batch_size / mean wall seconds.
A gpu_event_ms much smaller than wall_ms means the GPU is idle most of
the iteration (host overhead or data starvation), not that it is fast.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mpspulse", description="Apple Silicon MPS profiling and ETA")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("profile", help="profile a training step")
    pr.add_argument(
        "--model", required=True, help="resnet50|vit_b16|unet_tiny or script:module:function"
    )
    pr.add_argument("--batch-size", type=int, default=8)
    pr.add_argument("--iters", type=int, default=30)
    pr.add_argument("--warmup", type=int, default=5)
    pr.add_argument("--json", default=None, help="write report JSON here")
    pr.add_argument("--explain", action="store_true", help="print GPU-vs-wall explanation")
    pr.add_argument(
        "--dry-run", action="store_true", help="emit schema-shaped fixture without torch"
    )
    pr.add_argument(
        "--isolated",
        action="store_true",
        default=None,
        help="run measurement in a subprocess (default: on for script: targets, off for builtins)",
    )
    return p


def _load_script_target(spec: str):
    """Import `module:function` from a `script:module:function` spec."""
    _, _, dotted = spec.partition(":")
    mod_name, _, fn_name = dotted.rpartition(":")
    if not mod_name or not fn_name:
        raise ValueError(f"bad script spec {spec!r}; want script:module:function")
    module = importlib.import_module(mod_name)
    try:
        return getattr(module, fn_name)
    except AttributeError:
        raise ValueError(f"{mod_name} has no function {fn_name!r}") from None


def _memory_snapshot() -> dict:
    import torch

    return {
        "current_allocated": torch.mps.current_allocated_memory(),
        "driver_allocated": torch.mps.driver_allocated_memory(),
        "recommended_max": torch.mps.recommended_max_memory(),
    }


def run_profile(args: argparse.Namespace) -> dict:
    """Execute the profile measurement in-process; return the report dict."""
    from mpspulse.loop import loop_report_to_dict, profile_loop
    from mpspulse.models import MODELS, input_shape, make_model

    if args.model.startswith("script:"):
        target = _load_script_target(args.model)
        _require_mps_here()

        def step_fn(_batch):
            target()

        report = profile_loop(
            iter([None] * (args.warmup + args.iters)),
            step_fn,
            model=args.model,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iters=args.iters,
        )
    elif args.model in MODELS:
        import torch

        _require_mps_here()
        model = make_model(args.model).to("mps")
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        shape = input_shape(args.model, args.batch_size)
        batches = [
            torch.randn(*shape[1:], device="mps")
            .unsqueeze(0)
            .expand(args.batch_size, *shape[1:])
            .contiguous()
            for _ in range(args.warmup + args.iters)
        ]
        targets = [
            torch.randint(
                0, 1000 if args.model != "unet_tiny" else 2, (args.batch_size,), device="mps"
            )
            if args.model != "unet_tiny"
            else torch.randint(0, 2, (args.batch_size, 128, 128), device="mps")
            for _ in range(args.warmup + args.iters)
        ]

        def step_fn(batch):
            x, y = batch
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

        report = profile_loop(
            iter(zip(batches, targets)),
            step_fn,
            model=args.model,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        raise ValueError(f"unknown model {args.model!r}; choose from {MODELS} or script:mod:fn")

    return loop_report_to_dict(report, device=_device_string(), memory=_memory_snapshot())


def _device_string() -> str:
    import torch

    return f"mps (torch {torch.__version__})"


def _require_mps_here() -> None:
    from mpspulse.timing import _require_mps

    _require_mps()


def cmd_profile(args: argparse.Namespace) -> int:
    from mpspulse.errors import MpsUnavailableError
    from mpspulse.schema import dry_run_fixture, validate_report

    if args.explain and not args.dry_run and args.json is None:
        print(EXPLAIN)
        return 0
    if args.dry_run:
        report = dry_run_fixture(model=args.model, batch_size=args.batch_size)
    else:
        isolated = args.isolated
        if isolated is None:
            isolated = args.model.startswith("script:")
        if isolated:
            argv = [
                sys.executable,
                "-m",
                "mpspulse.subproc_profile",
                args.model,
                str(args.batch_size),
                str(args.iters),
                str(args.warmup),
            ]
            proc = subprocess.run(argv, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stderr.strip() or "subprocess failed", file=sys.stderr)
                return EXIT_USAGE
            report = json.loads(proc.stdout)
        else:
            try:
                report = run_profile(args)
            except MpsUnavailableError as exc:
                print(f"mpspulse: {exc}", file=sys.stderr)
                return EXIT_USAGE
            except ValueError as exc:
                print(f"mpspulse: {exc}", file=sys.stderr)
                return EXIT_USAGE
    errors = validate_report(report)
    if errors:
        print(f"mpspulse: invalid report: {errors}", file=sys.stderr)
        return EXIT_USAGE
    text = json.dumps(report, indent=2)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    if args.explain:
        print(EXPLAIN, file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "profile":
        try:
            return cmd_profile(args)
        except ValueError as exc:
            print(f"mpspulse: {exc}", file=sys.stderr)
            return EXIT_USAGE
    raise AssertionError(f"unreachable: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
