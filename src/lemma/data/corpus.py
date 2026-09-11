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

from ..schemas import AttemptRow, DeclarationRow, MutantRow

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPORA = REPO_ROOT / "corpora"
MANIFEST = CORPORA / "MANIFEST.yml"


class CorpusError(RuntimeError):
    pass


def encoder_text(row) -> str:
    """The statement string the encoder sees: the compact readable form when
    present, else the pp.all canonical form (older corpora)."""
    return row.readable_pp or row.canonical_type


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise CorpusError(f"no corpus manifest at {MANIFEST}")
    return yaml.safe_load(MANIFEST.read_text())


def verify_shard(path: Path, cid: str) -> None:
    actual = blake3.blake3(path.read_bytes()).hexdigest()
    if actual != cid:
        raise CorpusError(f"{path.name}: content hash {actual} != manifest CID {cid}")


def iter_mutants(corpus: str = "mutants", verify: bool = True) -> Iterator[MutantRow]:
    """Phase-3 mutant records (quod mutant_extract.py output), verified per shard."""
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
                    m = json.loads(line)
                    yield MutantRow(parent_name=m["parent"], operator=m["operator"],
                                    mutated_type=m["mutated_readable"], lock=m["lock"],
                                    lock_changed=m["lock_changed"], elaborates=m["elaborates"],
                                    gate_verdict=m["gate_verdict"], module=m.get("module"))


def iter_attempts(corpus: str = "attempts", verify: bool = True) -> Iterator[AttemptRow]:
    """quod attempt records (scripts/attempt.py output), verified per shard."""
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
                    yield AttemptRow.model_validate(json.loads(line))


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
