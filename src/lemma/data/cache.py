"""Tokenize once into uint16 memmap shards; near-zero RSS at train time.

The 15GB RAM ceiling is the machine's binding constraint: statements are
tokenized a single time per (corpus, tokenizer) pair into flat id arrays with
an offset index, and dataloaders read views of the memmap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MAGIC = "lemma-cache-v1"


def build(statements: list[str], tokenizer, out_dir: Path, name: str,
          max_seq: int = 512) -> dict:
    """Encode statements -> {name}.ids.npy / {name}.idx.npy + a stats dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_all: list[np.ndarray] = []
    offsets = [0]
    truncated = 0
    for s in statements:
        ids = tokenizer.encode(s).ids
        if len(ids) > max_seq:
            ids = ids[:max_seq]
            truncated += 1
        ids_all.append(np.asarray(ids, dtype=np.uint16))
        offsets.append(offsets[-1] + len(ids))
    flat = np.concatenate(ids_all) if ids_all else np.empty(0, dtype=np.uint16)
    np.save(out_dir / f"{name}.ids.npy", flat)
    np.save(out_dir / f"{name}.idx.npy", np.asarray(offsets, dtype=np.int64))
    stats = {
        "magic": MAGIC,
        "sequences": len(statements),
        "tokens": int(flat.size),
        "max_seq": max_seq,
        "truncated": truncated,
        "truncation_rate": round(truncated / max(len(statements), 1), 4),
    }
    (out_dir / f"{name}.stats.json").write_text(json.dumps(stats, indent=1))
    return stats


class CachedStatements:
    """Random access over a built cache without loading it into RAM."""

    def __init__(self, out_dir: Path, name: str):
        self.ids = np.load(out_dir / f"{name}.ids.npy", mmap_mode="r")
        self.idx = np.load(out_dir / f"{name}.idx.npy")
        stats = json.loads((out_dir / f"{name}.stats.json").read_text())
        if stats.get("magic") != MAGIC:
            raise ValueError(f"cache {name} has wrong magic: {stats.get('magic')}")
        self.stats = stats

    def __len__(self) -> int:
        return len(self.idx) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return np.asarray(self.ids[self.idx[i]:self.idx[i + 1]], dtype=np.int64)
