"""Pure unit tests for the sync scanner (no torch)."""

from pathlib import Path

from mpspulse.syncscan import scan_file, scan_path

FIXTURES = Path(__file__).parent / "fixtures"


def test_syncy_detections_with_lines():
    findings = scan_file(FIXTURES / "syncy.py")
    by_line = {f.line: f.pattern for f in findings}
    assert by_line[12] == ".item()"
    assert by_line[15] == ".cpu()"
    assert by_line[16] == ".numpy()"
    assert by_line[17] == "bool(tensor)"
    assert by_line[18] == "float(tensor)"
    assert by_line[19] == "torch.tensor(..., device)"
    assert all(f.in_hot_loop for f in findings)
    assert len(findings) == 6


def test_clean_has_no_findings():
    assert scan_file(FIXTURES / "clean.py") == []


def test_suggestions_present():
    for finding in scan_file(FIXTURES / "syncy.py"):
        assert finding.suggestion
        assert finding.to_dict()["file"].endswith("syncy.py")


def test_scan_directory():
    findings = scan_path(FIXTURES)
    assert len(findings) == 6
    assert {f.file for f in findings} == {str(FIXTURES / "syncy.py")}
