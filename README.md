# UI Reinforcement Learning

An attempt to RL fine-tune Computer Use models


## How to run

```bash
# 1. Setup kubctl
gcloud container clusters get-credentials simple-data-entry-cluster --region=europe-north2
 
# 2. Make sure the ui-verifiers proxy server is deployed
(cd /path/to/ui-verifiers/proxy; make deploy)

# 3. Find out public/external IP of proxy server
kubectl get services

# 4. Run demo script
uv run -m ui_rl.main <proxy ip>
```

Start vLLM model host:
uv run -m vllm.entrypoints.openai.api_server --model ByteDance-Seed/UI-TARS-1.5-7B --limit-mm-per-prompt '{"image":10, "video":0}'