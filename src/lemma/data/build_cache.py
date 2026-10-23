"""Build tokenized memmap caches per split from the declarations corpus.

    uv run python -m lemma.data.build_cache \
        --tokenizer runs/tokenizer/lean-unigram-32k.json --out corpora/cache

Reads the verified corpus, assigns leakage-safe splits (splits.py), tokenizes
the readable statement of each declaration once into uint16 memmap shards
(cache.py), and writes a split manifest. S2/S3/S4 read `declarations-<split>`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from . import splits
from .cache import build
from .corpus import encoder_text, iter_rows

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "corpora" / "cache")
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--rule", default=splits.CURRENT_RULE, choices=sorted(splits.ASSIGNERS),
                    help="split rule; v1 only to rebuild a legacy cache (it can split a module, #6)")
    ap.add_argument("--override-l12", action="store_true",
                    help="build a v2 split despite the L-12 giant-component guard (recorded in the manifest)")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(args.tokenizer))
    rows = list(iter_rows("declarations"))
    if args.rule == splits.RULE_V2 and not args.override_l12:
        try:
            splits.component_guard(rows)
        except ValueError as e:
            raise SystemExit(str(e))
    assignment = splits.assign_under(args.rule, rows)

    by_split: dict[str, list[str]] = {s: [] for s in splits.SPLITS}
    for r in rows:
        by_split[assignment[r.name]].append(encoder_text(r))

    report = {"tokenizer": args.tokenizer.name, "max_seq": args.max_seq, "splits": {}}
    for split, texts in by_split.items():
        stats = build(texts, tok, args.out, f"declarations-{split}", max_seq=args.max_seq)
        report["splits"][split] = stats
        print(f"{split}: {stats['sequences']} seqs, {stats['tokens']} tokens, "
              f"trunc {stats['truncation_rate']}")

    manifest = splits.manifest(rows, args.rule)
    if manifest["assignment"] != assignment:
        raise SystemExit("split manifest and cache assignment disagree")
    if args.override_l12:
        manifest["l12_override"] = True
    (args.out / "split-manifest.json").write_text(json.dumps(manifest, indent=1))
    (args.out / "cache-report.json").write_text(json.dumps(report, indent=1))
    print(f"split dep-jaccard(train,test)={manifest['train_test_dep_jaccard']} "
          f"counts={manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
