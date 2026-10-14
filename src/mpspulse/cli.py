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

    an = sub.add_parser("analyze", help="classify bottlenecks in a profile report")
    an.add_argument("--input", required=True, help="profile JSON from `profile`")
    an.add_argument(
        "--deep",
        action="store_true",
        help="also capture torch.mps.profiler + metalperftrace sample",
    )
    an.add_argument(
        "--model", default=None, help="model to run under --deep (default: report's model)"
    )
    an.add_argument("--batch-size", type=int, default=None)
    an.add_argument("--json", default=None, help="write analysis JSON here")

    sc = sub.add_parser("sync-scan", help="scan a file for host-device sync patterns")
    sc.add_argument("path", help="Python file or directory to scan (read-only)")
    sc.add_argument("--json", default=None, help="write findings JSON here")

    gt = sub.add_parser("gputrace", help="capture a .gputrace of a command (opt-in)")
    gt.add_argument("--output", default="trace.gputrace", help="trace output path")
    gt.add_argument(
        "--count", type=int, default=3, help="command buffers to capture (fewer = smaller trace)"
    )
    gt.add_argument(
        "--browse", action="store_true", help="attach a non-interactive gpudebug summary"
    )
    gt.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="command to run after `--`, e.g. gputrace -- python train.py",
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


def cmd_analyze(args: argparse.Namespace) -> int:
    from mpspulse.analyze import analyze

    try:
        with open(args.input) as fh:
            report = json.load(fh)
        result = analyze(report)
    except (OSError, ValueError, KeyError) as exc:
        print(f"mpspulse: cannot analyze {args.input}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.deep:
        result.deep = _deep_sample(args, report)
    text = json.dumps(result.to_dict(), indent=2)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    _print_analysis_human(result)
    return 0


def _deep_sample(args: argparse.Namespace, report: dict) -> dict:
    """Opt-in deep capture: profiler context then metalperftrace collect."""
    import shutil
    import tempfile

    deep: dict = {"warnings": []}
    model = args.model or report.get("model")
    batch_size = args.batch_size or report.get("batch_size", 8)
    if not model or model == "dry-run (no measurement)":
        deep["warnings"].append("deep: no runnable model; skipping capture")
        return deep
    try:
        from mpspulse.timing import _require_mps

        _require_mps()
        import torch

        ns = argparse.Namespace(model=model, batch_size=batch_size, iters=5, warmup=2)
        dataloader, step_fn, _ = _build_workload(ns)
        batches = [b for _, b in zip(range(7), dataloader)]
        for batch in batches[:2]:
            step_fn(batch)
        torch.mps.synchronize()
        with torch.mps.profiler.profile():
            for batch in batches[2:]:
                step_fn(batch)
        torch.mps.synchronize()
        deep["profiler"] = "torch.mps.profiler: 5 steps captured (signposts only)"
    except Exception as exc:  # noqa: BLE001 — base report must still emit
        deep["warnings"].append(f"deep: profiler capture failed: {exc}")
    if shutil.which("metalperftrace"):
        try:
            with tempfile.TemporaryDirectory(prefix="mps-deep-") as tmp:
                proc = subprocess.run(
                    ["metalperftrace", "collect", "--last", "30s", tmp],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                import glob as _glob

                traces = _glob.glob(f"{tmp}/*.atrc")
                if proc.returncode == 0 and traces:
                    ov = subprocess.run(
                        ["metalperftrace", "overview", "--json", traces[0]],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    deep["metalperftrace"] = ov.stdout[:2000] or "(empty overview)"
                else:
                    deep["warnings"].append(
                        f"deep: metalperftrace collect produced no trace (rc={proc.returncode})"
                    )
        except Exception as exc:  # noqa: BLE001 — best-effort capture
            deep["warnings"].append(f"deep: metalperftrace failed: {exc}")
    else:
        deep["warnings"].append("deep: metalperftrace not on PATH; skipping")
    return deep


def _print_analysis_human(result) -> None:
    print("## Summary", file=sys.stderr)
    print(result.summary, file=sys.stderr)
    print("## Metrics", file=sys.stderr)
    for key, value in result.metrics.items():
        print(f"- {key}: {value}", file=sys.stderr)
    if result.findings:
        print("## Findings", file=sys.stderr)
        for finding in result.findings:
            print(
                f"{finding.priority}. [{finding.class_}] {finding.metric}="
                f"{finding.value:.4f} (threshold {finding.threshold})",
                file=sys.stderr,
            )
            print(f"   {finding.recommendation}", file=sys.stderr)
    else:
        print("## Findings: none — clean bill.", file=sys.stderr)


def cmd_sync_scan(args: argparse.Namespace) -> int:
    from mpspulse.syncscan import scan_path

    try:
        findings = scan_path(args.path)
    except (OSError, SyntaxError) as exc:
        print(f"mpspulse: cannot scan {args.path}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    payload = [f.to_dict() for f in findings]
    text = json.dumps(payload, indent=2)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    for finding in findings:
        loop = "hot-loop" if finding.in_hot_loop else "cold"
        print(
            f"{finding.file}:{finding.line}: [{loop}] {finding.pattern} — {finding.suggestion}",
            file=sys.stderr,
        )
    if not findings:
        print("no host-device sync patterns found.", file=sys.stderr)
    return 0


def cmd_gputrace(args: argparse.Namespace) -> int:
    """Opt-in .gputrace capture. Never runs on the normal profile path."""
    import os
    import platform
    import shutil

    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        print("mpspulse: gputrace needs a command: gputrace -- <cmd...>", file=sys.stderr)
        return EXIT_USAGE
    if shutil.which("gpucapture") is None:
        print(
            "mpspulse: gpucapture not found (needs macOS 27+); base profile/estimate unaffected",
            file=sys.stderr,
        )
        return EXIT_USAGE
    macos_major = int(platform.mac_ver()[0].split(".")[0] or 0)
    if macos_major and macos_major < 27:
        print(f"mpspulse: gpucapture needs macOS 27+, this is {macos_major}", file=sys.stderr)
        return EXIT_USAGE
    env = {**os.environ, "MTL_CAPTURE_ENABLED": "1"}
    child = subprocess.Popen(cmd, env=env)
    try:
        boundary = _wait_for_boundary(child.pid, timeout_s=30)
        if boundary is None:
            print(
                "mpspulse: no capture boundary appeared (child may have "
                "exited too fast); run a longer workload",
                file=sys.stderr,
            )
            child.wait()
            return EXIT_USAGE
        cap = subprocess.run(
            [
                "gpucapture",
                "start",
                "--pid",
                str(child.pid),
                "--boundary",
                boundary,
                "--count",
                str(args.count),
                "--output",
                args.output,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if cap.returncode != 0:
            print(f"mpspulse: gpucapture failed: {cap.stderr.strip()[-300:]}", file=sys.stderr)
            child.wait()
            return EXIT_USAGE
    finally:
        if child.poll() is None:
            child.wait()
    print(f"trace: {args.output} (child rc={child.returncode})")
    if args.browse:
        _gpudebug_summary(args.output)
    return 0


def _wait_for_boundary(pid: int, timeout_s: float = 30) -> str | None:
    """Poll `gpucapture boundaries` until a Device/Queue boundary appears."""
    import time as _time

    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        proc = subprocess.run(
            ["gpucapture", "boundaries", "--pid", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in proc.stdout.splitlines():
            parts = line.split()
            # rows look like: `1  Device  Apple M2 Pro ...`
            if len(parts) >= 2 and parts[0].isdigit() and parts[1] in ("Device", "Queue"):
                return parts[0]
        _time.sleep(1.0)
    return None


def _gpudebug_summary(trace: str) -> None:
    import shutil

    if shutil.which("gpudebug") is None:
        print("mpspulse: gpudebug not found; skipping browse", file=sys.stderr)
        return
    print(
        "## gpudebug summary [experimental: graphics-oriented metrics on ML compute]",
        file=sys.stderr,
    )
    for browse_cmd in ("status", "list"):
        proc = subprocess.run(
            ["gpudebug", "-q", "--oneshot", "-t", trace, "-c", browse_cmd],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(f"$ gpudebug {browse_cmd}\n{proc.stdout[:1500]}", file=sys.stderr)


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
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "sync-scan":
        return cmd_sync_scan(args)
    if args.command == "gputrace":
        return cmd_gputrace(args)
    raise AssertionError(f"unreachable: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
