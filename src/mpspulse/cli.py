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

    es = sub.add_parser("estimate", help="estimate total training time")
    es.add_argument("--steps", type=int, required=True, help="total optimizer steps")
    es.add_argument("--epochs", type=int, default=1)
    es.add_argument("--mode", choices=["static", "benchmark", "live"], default="static")
    es.add_argument(
        "--step-ms",
        type=float,
        default=None,
        help="step time ms for static mode (or comma-separated samples)",
    )
    es.add_argument(
        "--model", default=None, help="model for benchmark mode (builtin or script:mod:fn)"
    )
    es.add_argument("--batch-size", type=int, default=8)
    es.add_argument("--iters", type=int, default=30, help="benchmark iters")
    es.add_argument("--warmup", type=int, default=5)
    es.add_argument("--grad-accum", type=int, default=1)
    es.add_argument("--startup-s", type=float, default=0.0)
    es.add_argument("--val-s", type=float, default=0.0, help="val overhead per epoch (s)")
    es.add_argument("--ckpt-s", type=float, default=0.0, help="checkpoint overhead per epoch (s)")
    es.add_argument("--json", default=None, help="write ETA JSON here")
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


def _build_workload(args: argparse.Namespace):
    """Build (dataloader, step_fn, model_name) for builtin or script targets."""
    from mpspulse.models import MODELS, input_shape, make_model

    if args.model.startswith("script:"):
        target = _load_script_target(args.model)
        _require_mps_here()

        def step_fn(_batch):
            target()

        return iter([None] * (args.warmup + args.iters)), step_fn, args.model
    if args.model in MODELS:
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
        n_cls = 1000 if args.model != "unet_tiny" else 2
        targets = [
            torch.randint(0, n_cls, (args.batch_size,), device="mps")
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

        return iter(zip(batches, targets)), step_fn, args.model
    raise ValueError(f"unknown model {args.model!r}; choose from {MODELS} or script:mod:fn")


def _profile_to_samples(args: argparse.Namespace) -> list[float]:
    """Profile once in throughput mode, return per-chunk wall-ms samples."""
    from mpspulse.loop import profile_loop

    dataloader, step_fn, _ = _build_workload(args)
    report = profile_loop(
        dataloader,
        step_fn,
        model=args.model,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
        gpu_events=False,
        throughput=True,
    )
    return report.wall_samples_ms


def _benchmark_wall_samples(args: argparse.Namespace) -> list[float]:
    if args.model.startswith("script:"):
        # script targets run isolated so caller state is untouched
        import subprocess as _subprocess

        argv = [
            sys.executable,
            "-m",
            "mpspulse.subproc_profile",
            args.model,
            str(args.batch_size),
            str(args.iters),
            str(args.warmup),
        ]
        proc = _subprocess.run(argv, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ValueError(proc.stderr.strip() or "subprocess failed")
        w = json.loads(proc.stdout)["wall_ms"]
        return [w["mean_ms"]] * max(w["n"], 1)
    return _profile_to_samples(args)


def run_profile(args: argparse.Namespace) -> dict:
    """Execute the profile measurement in-process; return the report dict."""
    from mpspulse.loop import loop_report_to_dict, profile_loop

    dataloader, step_fn, _ = _build_workload(args)
    report = profile_loop(
        dataloader,
        step_fn,
        model=args.model,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
    )
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


def cmd_estimate(args: argparse.Namespace) -> int:
    import hashlib

    from mpspulse.errors import MpsUnavailableError
    from mpspulse.eta import drift_warning, estimate, record_history

    overheads = {
        "startup_s": args.startup_s,
        "val_overhead_per_epoch_s": args.val_s,
        "ckpt_overhead_per_epoch_s": args.ckpt_s,
        "epochs": args.epochs,
        "grad_accum": args.grad_accum,
    }
    try:
        if args.mode == "static":
            if args.step_ms is None:
                print("mpspulse: static mode needs --step-ms", file=sys.stderr)
                return EXIT_USAGE
            samples = [float(x) for x in str(args.step_ms).split(",")]
            rep = estimate(args.steps, samples, source="static", **overheads)
        elif args.mode == "benchmark":
            if args.model is None:
                print("mpspulse: benchmark mode needs --model", file=sys.stderr)
                return EXIT_USAGE
            samples = _benchmark_wall_samples(args)
            rep = estimate(args.steps, samples, source="benchmark", **overheads)
            import statistics as _statistics

            p50 = _statistics.median(samples)
            cfg = hashlib.sha256(f"{args.model}|{args.batch_size}".encode()).hexdigest()[:16]
            drift = drift_warning(args.model, cfg, p50)
            if drift:
                rep.warnings.append(drift)
            record_history(args.model, cfg, p50, len(samples))
        else:  # live: step times on stdin, one float per line
            samples = [float(line) for line in sys.stdin if line.strip()]
            if not samples:
                print("mpspulse: live mode needs step times on stdin", file=sys.stderr)
                return EXIT_USAGE
            rep = estimate(args.steps, samples, source="live", **overheads)
    except MpsUnavailableError as exc:
        print(f"mpspulse: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as exc:
        print(f"mpspulse: {exc}", file=sys.stderr)
        return EXIT_USAGE
    text = json.dumps(rep.to_dict(), indent=2)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "profile":
        try:
            return cmd_profile(args)
        except ValueError as exc:
            print(f"mpspulse: {exc}", file=sys.stderr)
            return EXIT_USAGE
    if args.command == "estimate":
        return cmd_estimate(args)
    raise AssertionError(f"unreachable: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
