#!/bin/bash
cd "$(dirname "$0")/.."
RUN=s4v2-verdict-$(date +%Y%m%d-%H%M%S)
uv run python -u -m lemma.train_s4v2 --config configs/s4v2_verdict.yaml --run-id "$RUN" > "runs/$RUN.log" 2>&1
echo "train exit $?" >> "runs/$RUN.log"
uv run python -u -m lemma.evaluate_s4v2 --run "$RUN" > "runs/s4v2-eval-$RUN.log" 2>&1
echo "eval exit $?" >> "runs/s4v2-eval-$RUN.log"
