# Open Constitution Training Dataset

Training data for constitutional classifier probe experiments.

## Label Definition
- `label: 0` — Assistant response is safe/allowed (benign explanations, defensive guidance, safe refusals, authorized testing)
- `label: 1` — Assistant response materially helps harmful behavior

## Source
The current public-data path combines two Hugging Face safety datasets:

- a pairwise safety-labeled response dataset
- a single-response QA safety classification dataset

Convert it with:

```bash
uv run prepare-training-data \
  --output_path data/training_data.jsonl \
  --summary_path data/dataset_summary.json
```

## Topic Coverage
Topics are taken from each response's active harm category and normalized to snake case. Safe
responses with no active harm category use `safe`.

## Split Strategy
- Train: 80%
- Validation: 10%
- Test: 10%

Splits grouped by scenario family to minimize leakage.

## Conversion Rules
- Each pairwise source row emits one example for `response_0` and one for `response_1`
- Each single-response source row emits one example from `response`
- `label: 0` when the source safety label is `true`
- `label: 1` when the source safety label is `false`
- Empty responses or rows without a boolean safety label are skipped
- Source `train` prompts are deterministically split into train/validation by prompt
- The single-response source is capped by default with deterministic label/topic-diverse sampling
  so it augments rather than dominates the training set
- Duplicate prompt-response pairs are dropped by default
- All rows use the consistent 9-field schema

## Files
- `training_data.jsonl` — Converted training data
- `dataset_summary.json` — Distribution statistics for `training_data.jsonl`
