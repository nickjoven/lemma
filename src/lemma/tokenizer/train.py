"""S1: train and freeze the statement tokenizer.

Unigram, 24k vocab, byte-fallback, Lean-aware pre-tokenization. Trained once
on the pinned corpus and then content-addressed; a retrained tokenizer is a
new artifact with a new CID, never an overwrite. Token IDs therefore never
shift under a Mathlib pin change — unseen lexemes degrade to bytes.

Success gate (falsifiable, checked here, recorded by the caller):
  fertility <= 0.35 tokens/char on held-out statements, zero <unk>,
  exact round-trip on every sampled statement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

# Lean lexical atoms: keep unicode operators whole, split namespaced names at
# `.` so Nat.succ_le_iff shares pieces with Nat.succ, isolate brackets/binders.
LEAN_SPLIT = Regex(r"[A-Za-z_][A-Za-z0-9_']*|[0-9]+|\s+|.")


def build_trainer(vocab_size: int) -> trainers.UnigramTrainer:
    return trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<mask>", "<s>", "</s>"],
        unk_token=None,  # byte-fallback below makes <unk> unreachable
    )


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\."), behavior="isolated"),
            pre_tokenizers.Split(LEAN_SPLIT, behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def evaluate_tokenizer(tok: Tokenizer, held_out: list[str]) -> dict:
    total_tokens = total_chars = 0
    unk = 0
    roundtrip_fail = 0
    for s in held_out:
        enc = tok.encode(s)
        total_tokens += len(enc.ids)
        total_chars += len(s)
        unk += sum(1 for t in enc.tokens if t == "<unk>")
        if tok.decode(enc.ids) != s:
            roundtrip_fail += 1
    return {
        "fertility": round(total_tokens / max(total_chars, 1), 4),
        "unk": unk,
        "roundtrip_failures": roundtrip_fail,
        "held_out": len(held_out),
        "pass": (total_tokens / max(total_chars, 1)) <= 0.35 and unk == 0 and roundtrip_fail == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab-size", type=int, default=24_000)
    ap.add_argument("--holdout", type=int, default=2_000)
    args = ap.parse_args()

    from ..data.corpus import iter_rows

    statements = [r.canonical_type for r in iter_rows("declarations")]
    held_out, train = statements[: args.holdout], statements[args.holdout:]

    tok = build_tokenizer()
    tok.train_from_iterator(train, build_trainer(args.vocab_size))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.out))

    report = evaluate_tokenizer(Tokenizer.from_file(str(args.out)), held_out)
    print(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
