#!/bin/bash
cd "$(dirname "$0")/.."
uv run python -u -m lemma.w0_baselines --s2-run s2-mlm-20260909-141736 > runs/w0-baselines-3.log 2>&1
echo "baselines exit $?" >> runs/w0-baselines-3.log
RUN=w0-world-$(date +%Y%m%d-%H%M%S)
uv run python -u -m lemma.train_w0 train --config configs/w0_world.yaml --run-id "$RUN" > "runs/$RUN.log" 2>&1
echo "train exit $?" >> "runs/$RUN.log"
uv run python -u -m lemma.train_w0 evaluate --run "$RUN" > "runs/w0-eval-$RUN.log" 2>&1
echo "eval exit $?" >> "runs/w0-eval-$RUN.log"
