from contextlib import nullcontext
import os
from pathlib import Path
import shutil
from typing import Callable
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
import torch
import json
import time
import wandb
import queue
import torch.multiprocessing as mp
import plotly.graph_objects as go
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, AutoProcessor
from torch.utils.data import IterableDataset, DataLoader
from transformers.trainer_utils import is_main_process
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from rollout_worker import RolloutBatchRequest, rollout_worker
from simple_data_entry import SimpleDataEntryTaskSpec
from launch_vllm import DEFAULT_VLLM_ARGS, DEFAULT_VLLM_MOUNTS
from utils import Qwen2_5_VLCollate, get_rollout_result

from ui_rl.models.uitars15.dataset import UITARS15_RolloutDataset
from ui_rl.runner import FixedStrategy


def main(
    vllm_gpus: list[int],
    max_parallel_rollouts: int,
    vllm_mounts: list[str],
    vllm_args: list[str],
    lora_adapter: Path | None,
    rollouts_dir: Path,
    checkpoint_dir: Path,
):
    # Disable as we use multi-worker dataloader 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    base_model_name = "ByteDance-Seed/UI-TARS-1.5-7B"

    accelerator = Accelerator()

    if accelerator.is_main_process:
        wandb.init(project="ui-rl", group="rl")

    rollout_worker_fn = rollout_worker if accelerator.is_main_process else nullcontext()
    with rollout_worker_fn(
        gpus=vllm_gpus,
        model_name=base_model_name,
        max_parallel_rollouts=max_parallel_rollouts,
        vllm_mounts=vllm_mounts,
        vllm_args=vllm_args,
    ) as task_queue:

        # Load model
        processor = AutoProcessor.from_pretrained(base_model_name)
        collator = Qwen2_5_VLCollate(processor)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        if lora_adapter is not None:
            # Continue training from an existing LoRA adapter
            print(f"Loading LoRA adapter from: {lora_adapter}")
            model = PeftModel.from_pretrained(model, lora_adapter, is_trainable=True)
        else:
            # Start fresh with new LoRA adapter
            lora_config = LoraConfig(
                r=64,
                lora_alpha=64,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)

        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        optimizer_path = lora_adapter / "optimizer.pt"
        if lora_adapter is not None and optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path))

        model, optimizer = accelerator.prepare(model, optimizer)

        # Loop:
        #   Request rollout batch
        #   Load dataset
        #   Accumulate and step

        step = 0
        while True:

            # Start generation of new rollout batch
            if accelerator.is_main_process:
                if rollouts_dir.exists():
                    shutil.rmtree(rollouts_dir)
                strategy = FixedStrategy([SimpleDataEntryTaskSpec(rows=[i]) for i in list(range(2, 8) * 3)])
                rollout_batch_request = RolloutBatchRequest(
                    output_dir=rollouts_dir,
                    rollout_strategy=strategy,
                    lora_name=f"step_{step}",
                    lora_path=lora_adapter
                )
                task_queue.put(rollout_batch_request)


            # Start reading rollout data and accumulate gradients
            counter = mp.Value("i", 0)
            ds = LiveRolloutDataset(rollouts_dir, processor, reward_fn, counter, accelerator.process_index, accelerator.num_processes)
            dl = DataLoader(ds, batch_size=1, collate_fn=collator, num_workers=1)

            try:
                batch = next(dl)
            except StopIteration:
                raise RuntimeError("Did not receive any rollouts")

            while True:
                try:
                    # Try to fetch the NEXT batch immediately
                    next_batch = next(dl)
                    is_last_batch = False
                except StopIteration:
                    is_last_batch = True

                # Determine context: Sync only if it is the last batch
                context = accelerator.no_sync(model) if not is_last_batch else nullcontext()
                with context:
                    outputs = model(**batch)
                    # Note: Assumes batch_size == 1
                    loss = outputs.loss * batch["reward"][0]
                    
                    # Do NOT divide by steps here. 
                    # Accumulate the pure SUM of gradients.
                    accelerator.backward(loss) 
                    running_loss += loss.item()

                if is_last_batch:
                    break
                
                # Move to next
                batch = next_batch

            # --- LOOP FINISHED ---
            
            # Post-Hoc Gradient Scaling
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data /= counter.value

            # Step and Zero
            optimizer.step()
            optimizer.zero_grad()

            print(f"Update complete. Average Loss: {running_loss / counter.value}")

            step += 1

            if accelerator.is_main_process:
                # Log to wandb
                n_success, n_tot = get_rollout_result(rollouts_dir)
                success_rate = {row: n_success[row] / n_tot[row] for row in n_tot.keys()}
                fig = go.Figure(data=[go.Bar(x=list(success_rate.keys()), y=[success_rate[row] for row in success_rate.keys()])])
                wandb.log({
                    "success_rate": sum(success_rate.values()) / len(success_rate),
                    "row_success_rate": wandb.Plotly(fig)
                    "loss": running_loss / counter.value
                })

                # Save checkpoint for vLLM
                checkpoint_path = checkpoint_dir / f"step_{step}"
                peft_model: PeftModel = accelerator.unwrap_model(model)
                peft_model.save_pretrained(checkpoint_path, safe_serialization=True)
                lora_adapter = checkpoint_path


def reward_fn(rollout: UITARS15_RolloutDataset.Rollout):
    if set(rollout.progress["submitted_row_indices"]) == set([r-2 for r in rollout.task_spec["rows"]]):
        return 1.0
    else:
        return -1.0


class LiveRolloutDataset(IterableDataset):
    def __init__(
        self, 
        watch_dir: Path, 
        processor: Qwen2_5_VLProcessor, 
        reward_fn: Callable[[UITARS15_RolloutDataset.Rollout], float],
        global_len_counter: mp.Value,
        rank: int = 0,
        world_size: int = 1
    ):
        self.watch_dir = watch_dir
        self.processor = processor
        self.reward_fn = reward_fn
        self.global_len_counter = global_len_counter
        self.rank = rank
        self.world_size = world_size
        self.global_len = 0

    def __iter__(self):
        file_queue = queue.Queue[Path]()
        event_handler = FileQueueHandler(file_queue)

        for filepath in self.watch_dir.glob("*.json"):
            file_queue.put(filepath)

        if (done_file := Path(self.watch_dir, ".done")).exists():
            file_queue.put(done_file)

        observer = Observer()
        observer.schedule(event_handler, str(self.watch_dir), recursive=False)
        observer.start()
        
        print(f"(rank {self.rank}) Worker {torch.utils.data.get_worker_info().id} started watching {self.watch_dir}")

        try:
            while True:
                filepath = file_queue.get()
                if filepath.name == ".done":
                    print(f"(rank {self.rank}) Received .done, exiting...")
                    break
                elif filepath.name.endswith(".json"):
                    self._wait_for_file_completion(filepath)
                    print(f"(rank {self.rank}) process file: {filepath}")
                    rollout_ds = UITARS15_RolloutDataset(self.processor, str(filepath), self.reward_fn)
                    self.global_len_counter.value += len(rollout_ds)
                    rank_offset = hash(filepath.name) % self.world_size
                    for i, ex in enumerate(rollout_ds):
                        # Ensure only one rank gets each example
                        if (i + rank_offset) % self.world_size == self.rank:
                            yield ex
        finally:
            # Clean up thread when the worker dies
            observer.stop()
            observer.join()

    @staticmethod
    def _wait_for_file_completion(filepath):
        while True:
            try:
                with open(filepath, 'r') as f:
                    json.load(f) # Will crash if file ends abruptly
                    break
            except json.JSONDecodeError:
                time.sleep(0.5)
                continue

class FileQueueHandler(FileSystemEventHandler):
    def __init__(self, q: queue.Queue):
        self.q = q

    def on_created(self, event):
        if not event.is_directory:
            self.q.put(Path(event.src_path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-gpus", nargs="+", type=int)
    parser.add_argument("--max-parallel-rollouts", type=int)
    parser.add_argument("--vllm-mount", nargs="+")
    parser.add_argument("--lora-adapter", type=Path)
    parser.add_argument("--rollout-dir", type=Path, default=Path("/tmp/rollouts"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/tmp/checkpoints"))

    args, vllm_args = parser.parse_known_args()
    args.vllm_mounts += DEFAULT_VLLM_MOUNTS
    main(**vars(args), vllm_args=DEFAULT_VLLM_ARGS + vllm_args)