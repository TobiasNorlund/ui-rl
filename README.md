# UI Reinforcement Learning

An attempt to RL fine-tune Computer Use models


## How to run

```python

# During training, generate a batch of rollouts:
rollouts = cua_rollout(
    model=..., 
    reward_fn=...,
    new_session_fn=...
    n=...
)

# In cua_rollout, we create n k10n pods, each represented as a CUASession
# 
```