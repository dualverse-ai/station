"""Artifact-only verdict command used by the Research Center auditor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def submit_audit(eval_id: str, verdict: str, *, research_root: str | None = None) -> int:
    eval_id = str(eval_id).strip()
    verdict = str(verdict).strip().lower()
    if not eval_id or verdict not in {"pass", "fail"}:
        print("usage: submit_audit.sh <evaluation_id> pass|fail", file=sys.stderr)
        return 2
    root = Path(research_root or os.environ.get("STATION_RESEARCH_ROOT") or Path.cwd())
    audit_dir = root / "storage" / "audit"
    report_path = audit_dir / f"{eval_id}.md"
    if not report_path.is_file() or not report_path.read_text(encoding="utf-8").strip():
        print(f"auditor report is missing or empty: {report_path}", file=sys.stderr)
        return 1
    audit_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = audit_dir / f"{eval_id}.verdict"
    tmp_path = verdict_path.with_suffix(verdict_path.suffix + ".tmp")
    tmp_path.write_text(verdict + "\n", encoding="utf-8")
    os.replace(tmp_path, verdict_path)
    print(f"AUDIT_VERDICT: {verdict}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Submit an immutable Research Center audit verdict")
    parser.add_argument("eval_id")
    parser.add_argument("verdict", choices=("pass", "fail"))
    args = parser.parse_args(argv)
    return submit_audit(args.eval_id, args.verdict)


if __name__ == "__main__":
    raise SystemExit(main())
