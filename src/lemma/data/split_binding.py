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
  2. The run records none (every run sealed before #10): the rule must be
     PINNED, in configs/split_pins.yml (reviewed, provenance-covered) or by
     an explicit expected-rule argument. Anything else is refused. The
     assignment is rebuilt under the pinned rule; binding =
     "pinned_rule_reconstructed". This is weaker than (1): the pin says which
     code computed the split (by commit), and nothing the run sealed can
     confirm the rebuilt rows; callers that have more evidence (evaluate_s2's
     cache report) check it, and the binding is recorded with every number.

In both cases the rows must be the corpus the run trained on: the corpus
MANIFEST's source CID must be the run's recorded corpus_cids[0], every shard
is verified, and for the order-dependent RULE_V1 the shard order must be the
one the pin records (declarations_shards_blake3, from corpora/MANIFEST.yml at
the pinned commit), since in-shard order is fixed by the shard CIDs. An
explicit expected rule carries no shard order, so it cannot stand in for a v1
pin. A v1 reconstruction is therefore "the split rebuilt under the pinned
rule over the verified recorded corpus", not a split the run sealed.
Entrypoints call resolve_corpus, which reads the rows and their identity
together; resolve() trusts its caller that `rows` are the corpus `corpus`
names.

In both cases the resolved split is sealed as a certificate and its CID is
returned, so a downstream run records it as its own split_manifest_cid.
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


@dataclass
class Binding:
    assignment: dict[str, str]
    rule_id: str
    certificate_cid: str
    binding: str                  # "train_record_cid" | "pinned_rule_reconstructed"
    pin_source: str | None = None  # where the rule pin came from, when binding is a pin

    def disclosure(self) -> dict:
        return {"split_manifest_cid": self.certificate_cid, "split_rule": self.rule_id,
                "split_manifest_binding": self.binding, "split_pin_source": self.pin_source}


def load_pins(path: Path | None = None) -> dict:
    path = path or PINS_PATH
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text()) or {}).get("runs", {}) or {}


def put_certificate(cert: dict) -> str:
    # compact and key-sorted: the bytes are canonical for the content
    return ledger.ket_put(json.dumps(cert, sort_keys=True, separators=(",", ":")))


def check_corpus(run: ledger.RunRecord, corpus: dict) -> None:
    """The rows' corpus must be the one the run recorded (corpus_cids[0] is
    the declarations corpus, inherited from S2 by every later stage)."""
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
        return Binding(assignment, rule, recorded, "train_record_cid")

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
    cid = put_certificate(splits.certificate(assignment, rule))
    return Binding(assignment, rule, cid, "pinned_rule_reconstructed",
                   f"configs/{pins_path.name}" if pinned is not None else "explicit")


def resolve_corpus(run: ledger.RunRecord, *, expect_rule: str | None = None):
    """(rows, Binding) over the verified declarations corpus, in iter_rows
    order, tied to the corpus the run recorded."""
    from . import corpus
    rows, identity = corpus.read_declarations("declarations")
    return rows, resolve(run, rows, corpus=identity, expect_rule=expect_rule)
