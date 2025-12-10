# Task: Simple Data Entry

Desktop level task with two opened browser windows side-by-side, a Google Sheet with some data on the left and a Google Form on the right.
The task is to submit data from the spreadsheet into the form.

## Task variations

TODO: Add details

## Train

```bash
uv run train.py \
  --rollouts <(find ../../data/rollouts/{20251118_205148,20251119_124456,20251119_223016,20251128_132027,20251129_091728,20251129_101412,20251208_annotated}/ -name "*_success.json") \
  --num-test-rollouts 32 \
  --grad-accumulation-steps 8 \
  --eval-checkpoint-steps 1000 \
  --lora-adapter-path ../../data/checkpoints/20251128_224324/step_10000 \
  --task-error-rates it3_error_rates.pkl \
  --sampler-temperature 0.5
```