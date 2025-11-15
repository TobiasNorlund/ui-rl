# UI Reinforcement Learning

`ui-rl` implements Reinforcement Fine-tuning for Computer Use models. 
Allows training on verifiable UI tasks from [ui-verifiers](https://github.com/TobiasNorlund/ui-verifiers) to improve model reliability on targeted domains and tasks.

## How to run

```bash
# 1. Setup kubctl to connect
gcloud container clusters get-credentials simple-data-entry-cluster --region=europe-north2
 
# 2. Make sure the ui-verifiers proxy server is deployed
(cd /path/to/ui-verifiers/proxy; make deploy)

# 3. Find out public/external IP of proxy server (might need to wait a minute or so)
kubectl get services

# 4. Run demo script
uv run -m ui_rl.main <proxy ip>
```

Start vLLM model host:
VLLM_HTTP_TIMEOUT_KEEP_ALIVE=30 uv run vllm serve ByteDance-Seed/UI-TARS-1.5-7B --limit-mm-per-prompt '{"image":10, "video":0}' --max-num-seqs 8

CLUSTER_HOST=`kubectl get service proxy-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'`
MODEL_HOST=localhost
uv run ui_rl/generate_rollout_batch.py --cluster-host $CLUSTER_HOST:8000 --vllm-host $MODEL_HOST:8000 -n 20 -m 15