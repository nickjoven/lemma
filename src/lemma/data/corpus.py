"""JSONL corpus reader that refuses unverified shards.

A corpus arrives as shard files plus corpora/MANIFEST.yml mapping each shard
to the ket CID recorded by the producing quod run. Reading verifies bytes
against the CID (BLAKE3 of content) before parsing; a shard that does not
match its manifest entry is an error, not a warning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import blake3
import yaml

from ..schemas import DeclarationRow

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPORA = REPO_ROOT / "corpora"
MANIFEST = CORPORA / "MANIFEST.yml"


class CorpusError(RuntimeError):
    pass


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise CorpusError(f"no corpus manifest at {MANIFEST}")
    return yaml.safe_load(MANIFEST.read_text())


def verify_shard(path: Path, cid: str) -> None:
    actual = blake3.blake3(path.read_bytes()).hexdigest()
    if actual != cid:
        raise CorpusError(f"{path.name}: content hash {actual} != manifest CID {cid}")


def iter_rows(corpus: str = "declarations", verify: bool = True) -> Iterator[DeclarationRow]:
    man = load_manifest()
    shards = man.get("corpora", {}).get(corpus)
    if not shards:
        raise CorpusError(f"manifest has no corpus named {corpus!r}")
    for entry in shards:
        path = CORPORA / entry["file"]
        if verify:
            verify_shard(path, entry["cid"])
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield DeclarationRow.model_validate_json(line)
