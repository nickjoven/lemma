"""Typed records for corpora, runs, and evals.

A corpus row mirrors what quod's CorpusWalk emits per declaration at a pin.
A run record is the only citable form of a training or eval run: its metrics
exist exactly insofar as `metrics_cid` resolves in the project's ket store.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DeclarationRow(BaseModel):
    """One Mathlib declaration at a pin, as extracted by quod's corpus walker."""

    name: str
    module: str
    kind: Literal["theorem", "def", "axiom", "opaque", "inductive"]
    canonical_type: str          # pp.all — the lock's identity form
    readable_pp: Optional[str] = None  # default ppExpr — the compact encoder input
    lock: str
    hash_algo: Literal["blake3", "blake2b"] = "blake3"
    type_consts_std: list[str] = Field(default_factory=list)
    type_consts_custom: list[str] = Field(default_factory=list)
    proof_consts: list[str] = Field(default_factory=list)
    axioms: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    reduces_to_true: bool | Literal["timeout"] = False
    docstring: Optional[str] = None
    source_module: Optional[str] = None
    iff_lhs: Optional[str] = None
    iff_rhs: Optional[str] = None


class MutantRow(BaseModel):
    """A mutated statement with its gate-computed labels."""

    parent_name: str
    operator: str
    mutated_type: str
    lock: str
    lock_changed: bool
    elaborates: Optional[bool] = None
    gate_verdict: Optional[str] = None
    module: Optional[str] = None

    @property
    def op_class(self) -> str:
        """Operator class for prediction: hyp_del_k collapses to hyp_del; the
        proof-side operators (statement unchanged) collapse to proof_side."""
        if self.operator.startswith("hyp_del"):
            return "hyp_del"
        if self.operator in ("sorry_inject", "axiom_inject"):
            return "proof_side"
        return self.operator


class AttemptRow(BaseModel):
    """One line of quod's attempts.jsonl (scripts/attempt.py). The outcome is
    PROVER-RELATIVE: `no_proof_found` means the named prover, within its
    recorded budget, found nothing — never `refuted`, never "unknown"."""

    demonstrandum: str                       # theorem name (the parent, for mutant attempts)
    prover: str
    prover_config_cid: Optional[str] = None
    outcome: str                             # accepted | no_proof_found | kernel_rejected
    verdict: Optional[str] = None            # accepted | rejected: self-proof | rejected: extra axioms [...] | rejected: lean4checker | ...
    tactic: Optional[str] = None
    negated: bool = False
    mutant_operator: Optional[str] = None    # set for mutant attempts (CORPUS_MUTATE)
    mutant_lock: Optional[str] = None        # lock of the REBUILT mutant type; must equal the corpus mutant's lock
    selfproof_ok: Optional[bool] = None

    @property
    def gate_verdict(self) -> str:
        """Three-way label: proven | no_proof_found | rejected."""
        if self.verdict == "accepted":
            return "proven"
        if self.verdict and self.verdict.startswith("rejected"):
            return "rejected"
        return "no_proof_found"


class RunInputs(BaseModel):
    corpus_cids: list[str]
    split_manifest_cid: Optional[str] = None
    tokenizer_cid: Optional[str] = None
    config_cid: str
    init_checkpoint_cid: Optional[str] = None
    code_git: str
    env_lock_cid: Optional[str] = None
    mathlib_pin: str


class RunRecord(BaseModel):
    """One line of runs/ledger.jsonl. Append-only; never edited in place."""

    run_id: str
    kind: Literal["train", "eval", "calibration"]
    stage: str
    started: str
    wall_clock_s: Optional[float] = None
    status: Literal["running", "completed", "aborted", "calibration_failed"] = "running"
    inputs: RunInputs
    seed: int = 1337
    hardware: dict = Field(default_factory=dict)
    checkpoint_cids: list[str] = Field(default_factory=list)
    metrics_cid: Optional[str] = None
    calibration_record_cid: Optional[str] = None
