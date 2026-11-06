"""A real on-disk declarations corpus for tests: JSONL shards plus a
MANIFEST.yml naming each shard's BLAKE3, read through lemma.data.corpus
exactly as the real one is (verified, in manifest order)."""

import json

import blake3

from lemma.data import corpus

SOURCE = "fixture-quod-manifest"


def write_corpus(root, monkeypatch, shards, *, source=SOURCE, order=None):
    """Write `shards` (lists of DeclarationRow) under `root`, list them in
    MANIFEST.yml in `order` (default: as given), point lemma.data.corpus at
    it, and return the identity read_declarations will report."""
    (root / "declarations").mkdir(parents=True, exist_ok=True)
    entries = []
    for i, rows in enumerate(shards):
        f = root / "declarations" / f"declarations-{i:04d}.jsonl"
        f.write_text("".join(r.model_dump_json() + "\n" for r in rows))
        entries.append({"file": f"declarations/{f.name}", "cid": blake3.blake3(f.read_bytes()).hexdigest(),
                        "records": len(rows)})
    if order is not None:
        entries = [entries[i] for i in order]
    (root / "MANIFEST.yml").write_text(json.dumps({"source": {"manifest_cid": source},
                                                   "corpora": {"declarations": entries}}))
    monkeypatch.setattr(corpus, "CORPORA", root)
    monkeypatch.setattr(corpus, "MANIFEST", root / "MANIFEST.yml")
    return {"corpus": "declarations", "source_manifest_cid": source, "shards": len(entries),
            "shards_blake3": corpus.shard_order_blake3(entries)}
