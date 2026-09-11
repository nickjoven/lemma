"""Check a downloaded release asset against the ledger.

    uv run python -m lemma.verify_release <path> [--run RUN_ID]

Hashes the file (BLAKE3 of the raw bytes, the same check the corpus reader
applies to shards) and looks for a completed ledger record whose
`checkpoint_cids` contains that hash, or whose `inputs.tokenizer_cid` equals
it. No ket dependency: the ledger line is the claim, the hash is the check.

Exit 0 on a match, 1 when no ledger record cites the hash, 2 on usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import blake3

from . import ledger


def file_cid(path: Path) -> str:
    h = blake3.blake3()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def matches(cid: str, run_id: str | None = None) -> list[tuple[ledger.RunRecord, str]]:
    """(record, role) pairs whose sealed record cites `cid`; role is
    'checkpoint' or 'tokenizer'."""
    out = []
    for rec in ledger.read_ledger():
        if rec.status != "completed":
            continue
        if run_id and rec.run_id != run_id:
            continue
        if cid in rec.checkpoint_cids:
            out.append((rec, "checkpoint"))
        elif rec.inputs.tokenizer_cid == cid:
            out.append((rec, "tokenizer"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("--run", default=None, help="only accept a match from this run_id")
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    if not args.path.is_file():
        print(f"not a file: {args.path}", file=sys.stderr)
        return 2
    cid = file_cid(args.path)
    found = matches(cid, args.run)
    if not found:
        scope = f" for run {args.run}" if args.run else ""
        print(f"NO MATCH{scope}: the file hashes to {cid}; no completed ledger record cites it")
        return 1
    for rec, role in found:
        print(f"{rec.run_id}  stage {rec.stage}  status {rec.status}  ({role})")
    print(f"the ledger says {cid}; the file hashes to {cid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
