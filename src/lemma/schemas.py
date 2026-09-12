"""Typed records for corpora, runs, and evals.

A corpus row mirrors what quod's CorpusWalk emits per declaration at a pin.
A run record is the only citable form of a training or eval run: its metrics
exist exactly insofar as `metrics_cid` resolves in the project's ket store.
"""

from __future__ import annotations

from typing import Union, Literal, Optional

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
    dedup: bool = False                      # the mutant coincides with an existing theorem (same lock); a legitimate proof
    dedup_of: Optional[list[str]] = None
    source: str = "ladder"                   # ladder | search | adversarial
    predictor_cid: Optional[str] = None      # set when a learned predictor proposed the attempt

    @property
    def gate_verdict(self) -> str:
        """Three-way label: proven | no_proof_found | rejected."""
        if self.verdict == "accepted":
            return "proven"
        if self.verdict and self.verdict.startswith("rejected"):
            return "rejected"
        return "no_proof_found"


class TransitionRow(BaseModel):
    """One tactic step of one quod attempt (attempts/<run>/transitions/). Goals
    are identified by LOCK (the goal closed over its hypotheses, canonicalised
    like a statement); their text lives in the goals shards. `outcome` is
    closed | open | error | budget; a `budget` row is CENSORED (the heartbeat
    cap was hit), never a cost observation. `source` tells who produced the
    step (ladder | search | adversarial) and `predictor_cid` which model."""

    attempt_id: str
    demonstrandum: str
    demonstrandum_lock: Optional[str] = None
    mutant_operator: Optional[str] = None
    negated: bool = False
    prover: str
    prover_config_cid: Optional[str] = None
    source: str = "ladder"
    predictor_cid: Optional[str] = None
    pos: int
    kind: str                                # intros | rung
    tactic: str
    goal_before: str
    goal_after: Union[str, list[str]]        # "closed" | "error" | [locks]
    outcome: str
    err_class: str = ""
    err: str = ""
    heartbeats: int
    heartbeat_cap: int
    censored: bool = False
    wall_ms: int = 0
    on_accepted_path: bool = False
    attempt_outcome: str
    attempt_verdict: Optional[str] = None

    @property
    def progressed(self) -> bool:
        """The step changed the goal state without error (closed, or open goals
        differing from the input) — the rows the translation test uses."""
        return self.outcome == "closed" or (self.outcome == "open" and self.goal_after != [self.goal_before])


class GoalRow(BaseModel):
    lock: str
    readable: str
    readable_truncated: bool = False
    canonical: Optional[str] = None


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
