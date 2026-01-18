# Example project: Simple Data Entry

This project showcases how `ui-rl` can be used to fine-tune a Computer Use model to perfect a desktop-level UI task. This includes:

1. Constructing a containerized ephemeral task environment, see [env/](env/)
2. How to generate rollouts in the environment from a Computer Use model to evaluate its performance and success rate
3. How to improve model performance via Supervised Fine-Tuning (SFT) and Data Augmentation
4. How to further improve model performance via Reinforcement Learning from Verified Reward (RLVR)

`Simple Data Entry` is a desktop-level task with two opened browser windows side-by-side: A Google Sheet with some tabular data on the left and a Google Form on the right.
The task is to copy-paste some data (typically a specific row) from the spreadsheet into the form window, and then submit the form.

<img width="1738" height="1082" alt="Simple Data Entry task" src="https://github.com/user-attachments/assets/02fea119-5319-4cab-b803-21c87eebaee1" />



