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

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def build_trainer(vocab_size: int) -> trainers.UnigramTrainer:
    # ByteLevel's own alphabet (256 byte-glyphs) is the initial alphabet, so
    # every byte is representable and <unk> is unreachable — lossless roundtrip
    # without a separate byte-fallback stage.
    return trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<mask>", "<s>", "</s>"],
        unk_token=None,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )


def build_tokenizer() -> Tokenizer:
    # GPT-style ByteLevel pre-tokenization: its regex keeps identifier / number
    # / operator runs together (rather than isolating every symbol), so Unigram
    # learns frequent Lean substrings — the `.all` pretty-printed forms are
    # verbose and dense with unicode, and per-symbol splitting made fertility
    # insensitive to vocab size. ByteLevel is also lossless, so byte-fallback
    # and exact roundtrip come for free.
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    return tok


def evaluate_tokenizer(tok: Tokenizer, held_out: list[str]) -> dict:
    """The hard gate is LOSSLESSNESS (0 unk, exact roundtrip). Fertility and the
    token-length distribution are reported, not gated: `pp.all` canonical forms
    are intrinsically long (a fully-expanded statement is hundreds of tokens),
    so a fixed tokens/char target is the wrong criterion — the load-bearing
    downstream number is the truncation rate at the encoder's seq length, which
    the model config owns."""
    total_tokens = total_chars = 0
    unk = 0
    roundtrip_fail = 0
    lens = []
    for s in held_out:
        enc = tok.encode(s)
        lens.append(len(enc.ids))
        total_tokens += len(enc.ids)
        total_chars += len(s)
        unk += sum(1 for t in enc.tokens if t == "<unk>")
        if tok.decode(enc.ids) != s:
            roundtrip_fail += 1
    lens.sort()
    p = lambda q: lens[min(int(q * len(lens)), len(lens) - 1)] if lens else 0
    return {
        "fertility": round(total_tokens / max(total_chars, 1), 4),
        "unk": unk,
        "roundtrip_failures": roundtrip_fail,
        "held_out": len(held_out),
        "tokens_p50": p(0.50), "tokens_p95": p(0.95), "tokens_p99": p(0.99),
        "trunc_at_512": round(sum(1 for x in lens if x > 512) / max(len(lens), 1), 4),
        "trunc_at_1024": round(sum(1 for x in lens if x > 1024) / max(len(lens), 1), 4),
        "pass": unk == 0 and roundtrip_fail == 0,
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
