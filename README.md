# UI Reinforcement Learning

`ui-rl` is a library for fine-tuning Computer Use agent models.
It allows training on verifiable UI tasks to improve model reliability on targeted domains and tasks.

## Example: Generate rollouts for Simple Data Entry task

```bash
cd examples/simple_data_entry

# 1. Build Simple Data Entry docker image
(cd env && make build)

# 2. Start vLLM model host on each available GPUs (optionally with lora checkpoint preloaded)
#    Note: Requires docker compose
uv run launch_vllm.py --limit-mm-per-prompt '{"image":10, "video":0}' --max-num-seqs 8 \
    --extra-mount "$(realpath ../../)/data/checkpoints:/app/models" --enable-lora --max-lora-rank 64 --lora-modules step_2000=/app/models/20251210_195352/step_2000

# 3. Once vLLM is ready, start generating rollouts
uv run rollout_uitars15_docker.py --vllm-host localhost:8000 --strategy "nsuccessful(2-101;1;15;100)" --model-name step_2000 --max-parallel 120
```

## Example: Fine-tune model on successful rollouts (e.g. Rejection Sampling)
# 7. Run SFT on the successful rollouts (=rejection sampling)
uv run ui_rl/train.py \
    --rollouts `find data/rollouts/20251128_132027/ -name "*_success.json"` \
    --grad-accumulation-steps 8 \
    --eval-checkpoint-steps 1000 \
    --lora-adapter-path data/checkpoints/20251120_120610/step_10000
```
