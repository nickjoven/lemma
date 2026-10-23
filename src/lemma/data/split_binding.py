"""Which split a run trained against, by evidence, never by today's rule.

Review of #10: the evaluators recomputed splits.assign() at eval time, so the
#6 rule change silently moved rows into the test split of checkpoints that
had trained on them. Every evaluator, and every downstream trainer (which
inherits its init checkpoint's split, or its test rows were already seen in
that checkpoint's pretraining), resolves the split here instead:

  1. The run records a split_manifest_cid (a splits.certificate): the
     certificate is read from the store by CID, the assignment is rebuilt
     under the rule it names, and the rebuilt split must match its per-split
     name digests exactly. binding = "train_record_cid".
  2. The run records none (every run sealed before #10): the rule comes from
     a pin in configs/split_pins.yml (reviewed, provenance-covered), or from
     an explicit expected-rule argument (evaluate_s2 --expect-split-rule).
     With neither, the run is refused. The assignment is rebuilt under that
     rule; binding = "pinned_rule_reconstructed". This is weaker than (1):
     the pin says which code computed the split (by commit), and nothing the
     run sealed can confirm the rebuilt rows; callers that have more
     evidence (evaluate_s2's cache report) check it, and the binding is
     recorded with every number. An explicit rule on an unpinned run is an
     escape hatch, not evidence: it is taken on the caller's word, disclosed
     as pin_source "explicit", and only RULE_V2 can be given this way (v1
     needs the pin's shard order, below).

What ties the rows to the corpus the run trained on, exactly:
  - every path: corpora/MANIFEST.yml's source.manifest_cid must equal the
    run's corpus_cids[0] (a label train.py copied from its config), and
    every declarations shard must hash to the CID MANIFEST.yml lists for
    it. MANIFEST.yml itself is not hashed or pinned, and its shard CIDs are
    not checked against the quod manifest whose CID it names, so the source
    CID is a label that MANIFEST.yml asserts, not a verified identity.
  - a pinned v1 run, additionally: the shard order must equal the pin's
    declarations_shards_blake3 (from MANIFEST.yml at the pinned commit),
    which, with each shard verified, fixes the rows and their order.
  - a run with a certificate (1), additionally: split membership must match
    the certificate's per-split name digests. That binds which names are in
    which split, not the rows' content.
  - an explicit rule on an unpinned run: the label check above only, and the
    L-12 component guard (build_cache's) is not applied to the rebuild.
Only the declarations corpus is checked. The mutants, attempts,
transitions and goals corpora a downstream run reads come from the current
MANIFEST.yml and are not compared with the CIDs the run recorded.
A v1 reconstruction is therefore "the split rebuilt under the pinned rule
over the declarations shards MANIFEST.yml lists, in the pinned order", not a
split the run sealed. Entrypoints call resolve_corpus, which reads the rows
and their identity together; resolve() trusts its caller that `rows` are the
corpus `corpus` names.

In both cases the resolved split is sealed as a certificate and its CID is
returned, so a downstream run records it as its own split_manifest_cid. A
certificate made by a pinned rebuild carries that provenance (binding, pin
source, pinned run, corpus identity) inside its own bytes, so a run trained
on top of a pinned run, and that run's evaluation, reports the WEAKEST link
in the chain (pinned_rule_reconstructed, naming the pinned run), never
train_record_cid: the certificate binds the downstream run's eval split to
the downstream run's training exactly, but the init checkpoint's
pretraining split is still only a reconstruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .. import ledger
from . import splits
from .corpus import CorpusError

REPO_ROOT = Path(__file__).resolve().parents[3]
PINS_PATH = REPO_ROOT / "configs" / "split_pins.yml"


class SplitBindingError(CorpusError):
    pass


PINNED = "pinned_rule_reconstructed"


@dataclass
class Binding:
    assignment: dict[str, str]
    rule_id: str
    certificate_cid: str
    binding: str                  # the weakest link: "train_record_cid" | "pinned_rule_reconstructed"
    pin_source: str | None = None  # where the rule pin came from, when binding is a pin
    pinned_run: str | None = None  # the run whose split was reconstructed under that pin

    def disclosure(self) -> dict:
        return {"split_manifest_cid": self.certificate_cid, "split_rule": self.rule_id,
                "split_manifest_binding": self.binding, "split_pin_source": self.pin_source,
                "split_pinned_run": self.pinned_run}


def load_pins(path: Path | None = None) -> dict:
    path = path or PINS_PATH
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text()) or {}).get("runs", {}) or {}


def put_certificate(cert: dict) -> str:
    # compact and key-sorted: the bytes are canonical for the content
    return ledger.ket_put(json.dumps(cert, sort_keys=True, separators=(",", ":")))


def check_corpus(run: ledger.RunRecord, corpus: dict) -> None:
    """MANIFEST.yml's source CID must equal the run's corpus_cids[0] (the
    declarations corpus label, inherited from S2 by every later stage). This
    compares two labels; see the module docstring for what else binds the
    rows."""
    recorded = list(run.inputs.corpus_cids)
    source = corpus.get("source_manifest_cid")
    if not source or not recorded or recorded[0] != source:
        raise SplitBindingError(
            f"run {run.run_id} recorded declarations corpus {recorded[0] if recorded else None}, but the corpus "
            f"read now is {source}: its split cannot be rebuilt over a different corpus")


def resolve(run: ledger.RunRecord, rows, *, corpus: dict, expect_rule: str | None = None,
            pins_path: Path | None = None) -> Binding:
    """The split `run` trained against, rebuilt over `rows` (the declarations
    corpus in iter_rows order, whose identity is `corpus`, as
    corpus.read_declarations returns it) and checked as far as the record
    allows."""
    pins_path = pins_path or PINS_PATH
    check_corpus(run, corpus)
    recorded = run.inputs.split_manifest_cid
    if recorded:
        try:
            cert = json.loads(ledger.ket_get_verified(recorded))
        except ValueError as e:
            raise SplitBindingError(f"run {run.run_id}: split_manifest_cid {recorded} is not JSON: {e}") from e
        if not isinstance(cert, dict) or cert.get("format") != splits.CERT_FORMAT:
            raise SplitBindingError(f"run {run.run_id}: split_manifest_cid {recorded} is not a {splits.CERT_FORMAT}")
        rule = splits.rule_id(cert["rule_id"])
        if expect_rule is not None and splits.rule_id(expect_rule) != rule:
            raise SplitBindingError(f"run {run.run_id} trained under {rule}, not the expected {expect_rule}")
        assignment = splits.assign_under(rule, rows)
        why = splits.certificate_mismatch(cert, assignment)
        if why:
            raise SplitBindingError(f"run {run.run_id}: the split rebuilt under {rule} is not the one it "
                                    f"trained against ({why}); the corpus or the rule changed")
        prov = cert.get("provenance")
        if prov is None:
            return Binding(assignment, rule, recorded, "train_record_cid")
        if not isinstance(prov, dict) or prov.get("binding") != PINNED:
            raise SplitBindingError(f"run {run.run_id}: split_manifest_cid {recorded} has an unknown provenance {prov!r}")
        # trained on top of a pinned rebuild: the chain is only as strong as that rebuild
        return Binding(assignment, rule, recorded, PINNED, prov.get("pin_source"), prov.get("pinned_run"))

    pin = load_pins(pins_path).get(run.run_id)
    pinned = pin.get("split_rule") if isinstance(pin, dict) else None
    if pinned is None and expect_rule is None:
        raise SplitBindingError(
            f"run {run.run_id} records no split_manifest_cid and has no split-rule pin in {pins_path.name}; "
            "the split it trained against is unknown, so no held-out number is computed for it. Pin the rule "
            "(with its evidence) or pass the expected rule explicitly")
    if pinned is not None and expect_rule is not None and splits.rule_id(pinned) != splits.rule_id(expect_rule):
        raise SplitBindingError(f"run {run.run_id}: expected rule {expect_rule} contradicts the pin {pinned}")
    rule = splits.rule_id(pinned or expect_rule)
    if rule == splits.RULE_V1:
        # order-dependent: the rebuild is the trained split only over the rows in the order the build read them
        want = pin.get("declarations_shards_blake3") if pinned is not None else None
        if want is None:
            raise SplitBindingError(
                f"run {run.run_id}: {rule} depends on row order, and no pin records the shard order it was built "
                f"over (declarations_shards_blake3 in {pins_path.name}); an explicit rule cannot supply it")
        if want != corpus.get("shards_blake3"):
            raise SplitBindingError(
                f"run {run.run_id}: the corpus shard order {corpus.get('shards_blake3')} is not the pinned "
                f"{want}; {rule} would rebuild a different split")
    assignment = splits.assign_under(rule, rows)
    source = f"configs/{pins_path.name}" if pinned is not None else "explicit"
    cert = splits.certificate(assignment, rule)
    cert["provenance"] = {"binding": PINNED, "pin_source": source, "pinned_run": run.run_id,
                          "corpus": {k: corpus.get(k) for k in ("source_manifest_cid", "shards_blake3")}}
    return Binding(assignment, rule, put_certificate(cert), PINNED, source, run.run_id)


def resolve_corpus(run: ledger.RunRecord, *, expect_rule: str | None = None):
    """(rows, Binding) over the verified declarations corpus, in iter_rows
    order, tied to the corpus the run recorded."""
    from . import corpus
    rows, identity = corpus.read_declarations("declarations")
    return rows, resolve(run, rows, corpus=identity, expect_rule=expect_rule)
