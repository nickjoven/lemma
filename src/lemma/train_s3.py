"""S3: gloss-fidelity — align docstrings with their statements.

Shared encoder (warm-started from the S2 checkpoint) with two projection
heads, trained with symmetric InfoNCE over (docstring, readable statement)
pairs drawn from the declarations corpus. Splits are the module-level
leakage-safe assignment, so eval glosses come from held-out modules.

    uv run python -u -m lemma.train_s3 --config configs/s3_gloss.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from . import ledger
from .data import splits
from .data.corpus import encoder_text, iter_rows
from .models.encoder import Encoder, EncoderConfig
from .models.heads import ProjectionHead
from .objectives.contrastive import in_batch_accuracy, info_nce

REPO_ROOT = Path(__file__).resolve().parents[2]


class GlossModel(nn.Module):
    def __init__(self, enc: Encoder, out_dim: int):
        super().__init__()
        self.enc = enc
        self.proj_g = ProjectionHead(enc.cfg, out_dim)
        self.proj_s = ProjectionHead(enc.cfg, out_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))

    def encode(self, ids: torch.Tensor, which: str) -> torch.Tensor:
        z = self.enc.pool(ids)
        z = self.proj_g(z) if which == "g" else self.proj_s(z)
        return nn.functional.normalize(z, dim=-1)


def load_s2_encoder(ckpt: Path, vocab_size: int, max_seq: int) -> Encoder:
    cfg = EncoderConfig(vocab_size=vocab_size, max_seq=max_seq)
    enc = Encoder(cfg)
    sd = load_file(str(ckpt))
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    return enc


def gloss_pairs(split: str) -> list[tuple[str, str, str]]:
    """(docstring, statement, module) for declarations with a docstring in `split`."""
    rows = list(iter_rows("declarations", verify=False))
    assign = splits.assign(rows)
    return [(r.docstring, encoder_text(r), r.module) for r in rows
            if r.docstring and assign[r.name] == split]


def tokenize(tok: Tokenizer, texts: list[str], max_seq: int, pad_id: int) -> torch.Tensor:
    batch = [tok.encode(t).ids[:max_seq] for t in texts]
    width = max(len(b) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b)] = torch.tensor(b)
    return ids


def save_ckpt(model: GlossModel, path: Path) -> None:
    d = model.state_dict()
    save_file({k: v.contiguous() for k, v in d.items()}, str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    config_cid = ledger.ket_put(args.config.read_bytes())
    run_id = args.run_id or f"s3-gloss-{time.strftime('%Y%m%d-%H%M%S')}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 1337))

    tok = Tokenizer.from_file(cfg["tokenizer_path"])
    s2_rec = ledger.citable(cfg["s2_run"])
    s2_ckpt = REPO_ROOT / "runs" / "ckpts" / f"{cfg['s2_run']}.safetensors"
    enc = load_s2_encoder(s2_ckpt, tok.get_vocab_size(), cfg["max_seq"])
    model = GlossModel(enc, cfg.get("proj_dim", 256)).to(device)
    pad = enc.cfg.pad_id

    train = gloss_pairs("train")
    print(f"train gloss pairs: {len(train)}", flush=True)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": float(cfg.get("lr_encoder", 5e-5))},
        {"params": [*model.proj_g.parameters(), *model.proj_s.parameters(), model.logit_scale],
         "lr": float(cfg.get("lr_heads", 5e-4))},
    ], weight_decay=0.01, betas=(0.9, 0.98))

    rec = ledger.start_run(ledger.RunRecord(
        run_id=run_id, kind="train", stage="S3", started=ledger.now_iso(),
        inputs=ledger.RunInputs(corpus_cids=s2_rec.inputs.corpus_cids,
                                tokenizer_cid=s2_rec.inputs.tokenizer_cid,
                                config_cid=config_cid,
                                init_checkpoint_cid=s2_rec.checkpoint_cids[-1],
                                code_git=ledger.git_head(), mathlib_pin=s2_rec.inputs.mathlib_pin),
        seed=cfg.get("seed", 1337),
        hardware={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                  "torch": torch.__version__}))

    t0 = time.monotonic()
    rng = np.random.default_rng(cfg.get("seed", 1337))
    bs, steps, ckpt_every = cfg.get("batch_size", 128), cfg.get("steps", 8000), cfg.get("ckpt_every", 1000)
    ckpt_dir = REPO_ROOT / "runs" / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rolling = ckpt_dir / f"{run_id}-latest.safetensors"
    losses, accs = [], []
    model.train()
    for step in range(steps):
        if step > 0 and step % ckpt_every == 0:
            save_ckpt(model, rolling)
            print(f"step {step}: rolling checkpoint", flush=True)
        idx = rng.integers(0, len(train), size=bs)
        g_ids = tokenize(tok, [train[i][0] for i in idx], cfg["max_seq"], pad).to(device)
        s_ids = tokenize(tok, [train[i][1] for i in idx], cfg["max_seq"], pad).to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            g = model.encode(g_ids, "g")
            s = model.encode(s_ids, "s")
            loss = info_nce(g.float(), s.float(), model.logit_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            model.logit_scale.clamp_(0, float(np.log(100)))
        losses.append(loss.item())
        if step % 100 == 0:
            accs.append(in_batch_accuracy(g.float(), s.float()))
            print(f"step {step}: loss {np.mean(losses[-100:]):.4f} in-batch@1 {accs[-1]:.4f} "
                  f"scale {model.logit_scale.exp().item():.1f}", flush=True)

    ckpt = ckpt_dir / f"{run_id}.safetensors"
    save_ckpt(model, ckpt)
    cid = ledger.ket_put(ckpt.read_bytes())
    rec.checkpoint_cids.append(cid)
    rolling.unlink(missing_ok=True)
    gi = REPO_ROOT / ".gitignore"
    line = f".ket/cas/{cid}"
    if line not in gi.read_text().splitlines():
        with open(gi, "a") as f:
            f.write(f"{line}\n")

    ledger.finish_run(rec, metrics={
        "final_loss": float(np.mean(losses[-100:])),
        "in_batch_acc": float(accs[-1]) if accs else 0.0,
        "steps": steps, "train_pairs": len(train),
        "logit_scale": model.logit_scale.exp().item(),
    }, started_monotonic=t0)
    print(f"run {run_id} sealed; metrics_cid {rec.metrics_cid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
