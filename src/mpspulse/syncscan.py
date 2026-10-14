"""Static scan for host-device sync patterns (torch-free, read-only).

MPS port of NVIDIA perf-torch-sync-free detection: MPS has no
`set_sync_debug_mode` (verified absent on torch 2.14.1), so detection is a
read-only AST scan. Never rewrites code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PATTERNS: dict[str, str] = {
    ".item()": "defer to end of iteration, or keep on GPU with torch.where()",
    ".cpu()": ".to('cpu', non_blocking=True) only if CPU won't read it immediately",
    ".numpy()": "keep on GPU; move to CPU outside the hot loop",
    "bool(tensor)": "keep control flow on GPU with torch.where()",
    "float(tensor)": "defer scalar extraction out of the hot loop; coalesce metrics",
    "torch.tensor(..., device)": "create on CPU, then .to('mps', non_blocking=True)",
}


@dataclass
class SyncFinding:
    file: str
    line: int
    pattern: str
    in_hot_loop: bool
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "pattern": self.pattern,
            "in_hot_loop": self.in_hot_loop,
            "suggestion": self.suggestion,
        }


class _Visitor(ast.NodeVisitor):
    def __init__(self, file: str):
        self.file = file
        self.findings: list[SyncFinding] = []
        self._loop_depth = 0

    def _emit(self, node: ast.AST, pattern: str) -> None:
        self.findings.append(
            SyncFinding(self.file, node.lineno, pattern, self._loop_depth > 0, PATTERNS[pattern])
        )

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in ("item", "cpu", "numpy"):
                self._emit(node, f".{func.attr}()")
            elif func.attr == "tensor" and any(kw.arg == "device" for kw in node.keywords):
                self._emit(node, "torch.tensor(..., device)")
        elif isinstance(func, ast.Name):
            if func.id in ("bool", "float"):
                self._emit(node, f"{func.id}(tensor)")
            elif func.id == "tensor" and any(kw.arg == "device" for kw in node.keywords):
                self._emit(node, "torch.tensor(..., device)")
        self.generic_visit(node)


def scan_file(path: str | Path) -> list[SyncFinding]:
    """Scan one Python file; return sync findings sorted by line."""
    path = Path(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    visitor = _Visitor(str(path))
    visitor.visit(tree)
    return sorted(visitor.findings, key=lambda f: f.line)


def scan_path(path: str | Path) -> list[SyncFinding]:
    """Scan a file or directory (recursive *.py); sorted by file then line."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    findings: list[SyncFinding] = []
    for file in files:
        try:
            findings.extend(scan_file(file))
        except (SyntaxError, UnicodeDecodeError):
            continue
    return sorted(findings, key=lambda f: (f.file, f.line))
