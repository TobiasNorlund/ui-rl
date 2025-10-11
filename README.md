# UI Reinforcement Learning

An attempt to RL fine-tune Computer Use models


## How to run

```bash
# 1. Setup kubctl
gcloud container clusters get-credentials simple-data-entry-cluster --region=europe-north2
 
# 2. Deploy the ui-verifiers proxy server

# 3. Run demo script
uv run -m ui_rl.main
```