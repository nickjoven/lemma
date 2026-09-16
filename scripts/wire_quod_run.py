"""wire_quod_run.py <quod attempt run dir> [--attempts] [--transitions]

Copy a sealed quod attempt run's outputs into corpora/ and record them in
corpora/MANIFEST.yml BY CID: every shard's blake3 must equal the CID the quod
manifest recorded (a copy that does not hash to its CID is refused). Nothing
is typed by hand; the source block carries the quod run id, its manifest
CID (from the driver log line "manifest CID: ...") and prover config CID.

    uv run python scripts/wire_quod_run.py ~/code/quod/attempts/theorems-A-t1 --transitions
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import blake3
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPORA = REPO_ROOT / "corpora"
MANIFEST = CORPORA / "MANIFEST.yml"


def cid_of(p: Path) -> str:
    return blake3.blake3(p.read_bytes()).hexdigest()


def copy_verified(src: Path, dst: Path, cid: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    actual = cid_of(dst)
    if actual != cid:
        dst.unlink()
        sys.exit(f"refused: {src} hashes to {actual}, quod manifest says {cid}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--attempts", action="store_true")
    ap.add_argument("--transitions", action="store_true")
    ap.add_argument("--replace", action="store_true", help="replace the corpus entries instead of appending shards")
    ap.add_argument("--note", default="", help="a caveat recorded on the source block (e.g. a quod OPEN.yml item)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    mpath = sorted(p for p in glob.glob(str(run_dir / "manifest-*.json")) if "/manifest-pre-" not in p)[-1]  # the pre-run seal is not the manifest
    man = json.load(open(mpath))
    run_id = man["run_id"]
    log = run_dir / "driver.log"
    m = re.search(r"manifest CID: ([0-9a-f]{64})", log.read_text()) if log.exists() else None
    manifest_cid = m.group(1) if m else None
    if manifest_cid is None:
        sys.exit("no 'manifest CID' line in driver.log — was the run sealed with ket?")
    lm = yaml.safe_load(MANIFEST.read_text())
    lm.setdefault("corpora", {}); lm.setdefault("sources", {})
    src_block = {"run_id": run_id, "prover": man["prover"], "manifest_cid": manifest_cid,
                 "prover_config_cid": man.get("prover_config_cid"), "pins": man.get("pins"),
                 "regate_of": (man.get("regate_of") or {}).get("run_id")}
    if args.attempts:
        cid = man["attempts_cid"]
        dst = CORPORA / "attempts" / f"attempts-{run_id}.jsonl"
        copy_verified(run_dir / "attempts.jsonl", dst, cid)
        entry = {"file": str(dst.relative_to(CORPORA)), "cid": cid, "records": man["attempts"]}
        lm["corpora"]["attempts"] = [entry] if args.replace else [*lm["corpora"].get("attempts", []), entry]
        lm["sources"]["attempts"] = src_block
        print(f"attempts: {entry}")
    if args.transitions:
        tr = man["transitions"]
        # attempt.py writes one flat shard list; the batch driver (attempt_b.py, tiers B and L) writes one
        # list per round under r<N>/, and the copied file carries the round in its name.
        if "rounds" in tr:
            shards = [(f"r{r['round']}", s) for r in tr["rounds"] for s in r["shards"]]
            src_key = "transitions_" + man["prover"].lower().replace("-", "_")
            totals = {"n_transitions": tr["n_transitions"], "rounds": len(tr["rounds"]),
                      "n_goals": sum(r["n_goals"] for r in tr["rounds"]), "censored": sum(r["censored"] for r in tr["rounds"])}
        else:
            shards = [("", s) for s in tr["shards"]]
            src_key = "transitions"
            totals = {"n_transitions": tr["n_transitions"], "n_goals": tr["n_goals"], "censored": tr["censored"],
                      "accepted_path_fraction": tr["accepted_path_fraction"]}
        for kind in ("goals", "transitions"):
            entries = []
            for rnd, s in shards:
                if s["kind"] != kind or not s.get("cid"):
                    continue
                tag = f"{run_id}-{rnd}-" if rnd else f"{run_id}-"
                dst = CORPORA / kind / f"{tag}{Path(s['file']).name}"
                copy_verified(run_dir / rnd / s["file"], dst, s["cid"])
                entries.append({"file": str(dst.relative_to(CORPORA)), "cid": s["cid"], "records": s["records"]})
            lm["corpora"][kind] = entries if args.replace else [*lm["corpora"].get(kind, []), *entries]
            print(f"{kind}: {len(entries)} shard(s), {sum(e['records'] for e in entries)} records")
        lm["sources"][src_key] = {**src_block, **totals, **({"model": man["prover_config"]["model"]} if "prover_config" in man else {}),
                                  **({"note": args.note} if args.note else {})}
    MANIFEST.write_text(yaml.safe_dump(lm, sort_keys=False))
    print(f"MANIFEST.yml updated from {run_id} (manifest CID {manifest_cid[:12]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
