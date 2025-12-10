# UI Reinforcement Learning

`ui-rl` is a library for fine-tuning of Computer Use agent models.
It allows training on verifiable UI tasks to improve model reliability on targeted domains and tasks.

## How to run

Tested on a machine with one A100/H100 80GB

```bash
# 1. Setup kubctl to connect
gcloud container clusters get-credentials simple-data-entry-cluster --region=europe-north2
 
# 2. Make sure the ui-verifiers proxy server is deployed
(cd /path/to/ui-verifiers/proxy; make deploy)

# 3. Find out public/external IP of proxy server (might need to wait a minute or so)
kubectl get services

# 4. Start vLLM model host
VLLM_HTTP_TIMEOUT_KEEP_ALIVE=30 uv run vllm serve ByteDance-Seed/UI-TARS-1.5-7B --limit-mm-per-prompt '{"image":10, "video":0}' --max-num-seqs 8 \
    --enable-lora --max-lora-rank 64 --lora-modules `find data/checkpoints/20251120_120610/ -mindepth 1 -maxdepth 1 -type d -printf "%f=%p "`

# 5. Generate a batch of rollouts
CLUSTER_HOST=`kubectl get service proxy-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'`
MODEL_HOST=localhost
uv run ui_rl/rollout/generate_rollout_batch.py --cluster-host $CLUSTER_HOST:8000 --vllm-host $MODEL_HOST:8000 --strategy "nsuccess(2-101;1;100)" --model-name step_10000 --max-parallel 15

# 7. Run SFT on the successful rollouts (=rejection sampling)
uv run ui_rl/train.py \
    --rollouts `find data/rollouts/20251128_132027/ -name "*_success.json"` \
    --grad-accumulation-steps 8 \
    --eval-checkpoint-steps 1000 \
    --lora-adapter-path data/checkpoints/20251120_120610/step_10000
```
