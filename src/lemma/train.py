"""S2 training entrypoint: span-MLM over the statement cache.

Every run goes through the ledger: config is ket-put before compute, metrics
are ket-put at the end, or the run is not citable. bf16 autocast on the 4070;
torch.compile behind a config flag with a documented fallback.

    uv run python -m lemma.train --config configs/s2_mlm.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer

from . import ledger
from .data.cache import CachedStatements
from .models.encoder import Encoder, EncoderConfig
from .models.heads import MLMHead
from .objectives.mlm import masked_accuracy, mlm_loss, span_mask

REPO_ROOT = Path(__file__).resolve().parents[2]


def collate(batch: list[np.ndarray], pad_id: int, max_seq: int) -> torch.Tensor:
    width = min(max(len(b) for b in batch), max_seq)
    out = torch.full((len(batch), width), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        n = min(len(b), width)
        out[i, :n] = torch.from_numpy(b[:n])
    return out


def record_cache_split(cache_dir: Path) -> str:
    """Seal the certificate of the split the cache in `cache_dir` was built
    from; refuses a cache without its manifest, a manifest without a known
    rule id, or one whose counts are not the cache report's. A component-rule
    (v2) manifest must also pass the L-12 guard on the component statistics
    it discloses, or carry build_cache's l12_override, which then goes into
    the certificate (and so into the train record's split_manifest_cid)."""
    import json

    from .data import split_binding, splits
    split_manifest, report = cache_dir / "split-manifest.json", cache_dir / "cache-report.json"
    for f in (split_manifest, report):
        if not f.is_file():
            raise SystemExit(f"{f} is missing: a cache without its split manifest and report "
                             "cannot support a later held-out evaluation")
    manifest = json.loads(split_manifest.read_text())
    if "rule_id" not in manifest:
        raise SystemExit(f"{split_manifest} names no rule_id (built before #10); rebuild the cache")
    cert = splits.certificate(manifest["assignment"], manifest["rule_id"])
    if cert["rule_id"] == splits.RULE_V2:
        largest = (manifest.get("components") or {}).get("largest_component_rows")
        if largest is None:
            raise SystemExit(f"{split_manifest} discloses no component statistics; the L-12 guard cannot be checked")
        frac = largest / max(len(manifest["assignment"]), 1)
        if frac > splits.GIANT_COMPONENT_MAX_FRACTION and manifest.get("l12_override") is not True:
            raise SystemExit(f"{split_manifest}: the largest component holds {frac:.1%} of the rows, over the "
                             f"L-12 guard, and the manifest records no l12_override (OPEN.yml L-12)")
        if manifest.get("l12_override") is True:
            cert["l12_override"] = True
    want = {s: v["sequences"] for s, v in json.loads(report.read_text())["splits"].items()}
    if {s: cert["counts"].get(s, 0) for s in want} != want:
        raise SystemExit(f"split manifest counts {cert['counts']} differ from the cache report {want}")
    return split_binding.put_certificate(cert)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    src_state = ledger.preflight_source()  # refuse a dirty tree before compute; start_run checks it did not change since

    cfg = yaml.safe_load(args.config.read_text())
    config_cid = ledger.ket_put(args.config.read_bytes())
    run_id = args.run_id or f"s2-mlm-{time.strftime('%Y%m%d-%H%M%S')}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 1337))
    torch.set_float32_matmul_precision("high")

    tok = Tokenizer.from_file(cfg["tokenizer_path"])
    cache = CachedStatements(REPO_ROOT / cfg["cache_dir"], cfg["cache_name"])
    # the split this checkpoint trains against, by CID (lemma #5): a
    # splits.certificate (rule id + per-split name digests), which every
    # evaluator and downstream trainer verifies a rebuild against
    # (split_binding.resolve). Not the full manifest: at ~296k names that is
    # 15-20 MB per run in the git-tracked store, and the certificate pins the
    # same assignment exactly (review of #10).
    split_manifest_cid = record_cache_split(REPO_ROOT / cfg["cache_dir"])
    enc_cfg = EncoderConfig(vocab_size=tok.get_vocab_size(), **cfg.get("encoder", {}))
    model = Encoder(enc_cfg).to(device)
    head = MLMHead(model).to(device)
    if cfg.get("compile", True):
        try:
            model = torch.compile(model)
        except Exception as e:  # WSL2/triton papercuts: fall back, loudly
            print(f"torch.compile failed ({e}); continuing eager")
    # Dedup: MLMHead ties its decoder weight to the encoder embedding, so the
    # tied tensor appears in both parameter lists — pass each tensor once.
    params = list({id(p): p for p in [*model.parameters(), *head.parameters()]}.values())
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 3e-4)),
                            weight_decay=0.01, betas=(0.9, 0.98))

    rec = ledger.start_run(ledger.RunRecord(
        run_id=run_id, kind="train", stage="S2", started=ledger.now_iso(),
        inputs=ledger.RunInputs(
            corpus_cids=cfg.get("corpus_cids", []),
            split_manifest_cid=split_manifest_cid,
            tokenizer_cid=cfg.get("tokenizer_cid"),
            config_cid=config_cid, code_git=ledger.git_head(),
            mathlib_pin=cfg["mathlib_pin"]),
        seed=cfg.get("seed", 1337),
        hardware={"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                  "torch": torch.__version__}), preflight=src_state)

    t0 = time.monotonic()
    rng = np.random.default_rng(cfg.get("seed", 1337))
    batch_size = cfg.get("batch_size", 96)
    steps = cfg.get("steps", 10_000)
    ckpt_every = cfg.get("ckpt_every", 2000)
    mask_id = tok.token_to_id("<mask>")
    losses, accs = [], []
    ckpt_dir = REPO_ROOT / "runs" / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    def save_ckpt(path: Path) -> None:
        base = getattr(model, "_orig_mod", model)
        d = {**{f"encoder.{k}": v for k, v in base.state_dict().items()},
             **{f"head.{k}": v for k, v in head.state_dict().items()}}
        # head.decoder.weight IS encoder.embed.weight (tied); safetensors refuses
        # shared-memory tensors, and it is reconstructed by re-tying on load.
        # This crashed the first rolling checkpoint at step 2000 on 2026-09-09.
        d.pop("head.decoder.weight", None)
        save_file(d, str(path))

    # Rolling on-disk checkpoint. A 12h42m run was lost on 2026-09-09 because
    # the only save was at the final step; a kill (or reboot) must never cost
    # more than ckpt_every steps again. Only the FINAL checkpoint is ket-put —
    # periodic ones overwrite one file so disk/CAS don't fill with 30 copies.
    rolling = ckpt_dir / f"{run_id}-latest.safetensors"
    model.train()
    for step in range(steps):
        if step > 0 and step % ckpt_every == 0:
            save_ckpt(rolling)
            print(f"step {step}: rolling checkpoint -> {rolling.name}", flush=True)
        idx = rng.integers(0, len(cache), size=batch_size)
        ids = collate([cache[int(i)] for i in idx], enc_cfg.pad_id, enc_cfg.max_seq).to(device)
        corrupted, labels = span_mask(ids, enc_cfg.pad_id, mask_id, enc_cfg.vocab_size)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = head(model(corrupted))
            loss = mlm_loss(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            accs.append(masked_accuracy(logits.float(), labels))
            # flush: the run is detached with stdout redirected to a file, and
            # block-buffering would hide progress for thousands of steps
            print(f"step {step}: loss {np.mean(losses[-100:]):.4f} acc {accs[-1]:.4f}", flush=True)

    ckpt = ckpt_dir / f"{run_id}.safetensors"
    save_ckpt(ckpt)
    cid = ledger.ket_put(ckpt.read_bytes())
    rec.checkpoint_cids.append(cid)
    rolling.unlink(missing_ok=True)
    # .ket/ is git-tracked (federation model) but a 126 MB checkpoint blob
    # exceeds GitHub's 100 MB file limit and rejected the push on 2026-09-10.
    # The CID stays resolvable locally via `ket get` (provenance intact); the
    # bytes are kept out of git. Manifests and CIDs travel; weights do not.
    gi = REPO_ROOT / ".gitignore"
    line = f".ket/cas/{cid}"
    if line not in gi.read_text().splitlines():
        with open(gi, "a") as f:
            f.write(f"{line}\n")

    ledger.finish_run(rec, metrics={
        "final_loss": float(np.mean(losses[-100:])),
        "masked_acc": float(accs[-1]) if accs else 0.0,
        "steps": steps, "truncation_rate": cache.stats["truncation_rate"],
    }, started_monotonic=t0)
    print(f"run {run_id} sealed; metrics_cid {rec.metrics_cid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
