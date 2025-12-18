#!/usr/bin/env python3
"""
Migration script for rollout JSON files.

Transforms old completion format to new format:
- Old: {"context": [...], "completion": {...}, "logprobs": [...]}
- New: {"prompt_messages": [...indices...], "generated_message": index,
        "prompt_token_ids": [...], "generated_token_ids": [...]}
"""

import json
import argparse
from pathlib import Path
from copy import deepcopy
from typing import List, Dict
import shutil
from transformers import AutoProcessor


def tokenize_messages(processor, messages: List[Dict]) -> List[int]:
    """Tokenize messages using processor.apply_chat_template."""
    try:
        # Apply chat template to get token IDs
        token_ids = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None
        )
        # Convert to list if it's a tensor or array
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()

        # Remove batch dimension if present (apply_chat_template may add it)
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        return token_ids
    except Exception as e:
        print(f"Warning: Could not tokenize messages: {e}")
        # Return empty list as fallback
        return []


def tokenize_message_content(processor, message: Dict) -> List[int]:
    """Tokenize a single message's content."""
    try:
        content = message.get("content", "")
        if isinstance(content, list):
            # If content is a list (multimodal), extract text only
            text_content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
        else:
            text_content = content

        # Tokenize the text content
        token_ids = processor.tokenizer.encode(text_content, add_special_tokens=True) + [151645]  # add end of message token: <|im_end|>'
        return token_ids
    except Exception as e:
        print(f"Warning: Could not tokenize message content: {e}")
        return []


def migrate_completion(completion: Dict, messages: List[Dict], processor) -> Dict:
    """Migrate a single completion from old format to new format."""
    # Extract old fields (already indices, not message objects)
    prompt_message_indices = completion.get("context", [])  # List[int]
    generated_message_index = completion.get("completion")  # int
    # logprobs are dropped (not extracted)

    # Get actual messages from indices for tokenization
    prompt_messages = [messages[i] for i in prompt_message_indices]
    generated_message = messages[generated_message_index]

    # Tokenize to get token IDs
    prompt_token_ids = tokenize_messages(processor, prompt_messages)
    generated_token_ids = tokenize_message_content(processor, generated_message)

    # Return new format
    return {
        "prompt_token_ids": prompt_token_ids,
        "prompt_messages": prompt_message_indices,
        "generated_token_ids": generated_token_ids,
        "generated_message": generated_message_index,
    }


def migrate_rollout_file(filepath: Path, processor, backup: bool = True, dry_run: bool = False, output_path: Path = None):
    """Migrate a single rollout JSON file."""
    print(f"\nProcessing: {filepath}")

    # Load the JSON file
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Check if already migrated
    if data.get("completions") and len(data["completions"]) > 0:
        first_completion = data["completions"][0]
        if "prompt_token_ids" in first_completion and "prompt_messages" in first_completion:
            print(f"  ✓ Already migrated (has new format)")
            return

        if "context" not in first_completion and "completion" not in first_completion:
            print(f"  ✓ Already migrated or has different format")
            return

    # Get messages list
    messages: list = deepcopy(data.get("messages", []))
    if not messages:
        print(f"  ✗ No messages found in file")
        return
    
    # Preprocess messages
    for message in messages:
        # Convert "content" to list if str
        if type(message["content"]) == str:
            message["content"] = [{"type": "text", "text": message["content"]}]
        # Remove all keys except "content" or "role"
        keys_to_remove = [k for k in list(message.keys()) if k not in ("role", "content")]
        for k in keys_to_remove:
            del message[k]
        # Convert "image_url" -> "image"
        for block in message["content"]:
            if block["type"] == "image_url":
                block["type"] = "image"
                block["image"] = block["image_url"]["url"]
                del block["image_url"]

    # Migrate completions
    old_completions = data.get("completions", [])
    if not old_completions:
        print(f"  ✓ No completions to migrate")
        return

    print(f"  Migrating {len(old_completions)} completions...")
    new_completions = []

    for i, completion in enumerate(old_completions):
        try:
            new_completion = migrate_completion(completion, messages, processor)
            new_completions.append(new_completion)
            print(f"    Completion {i+1}/{len(old_completions)} ✓")
        except Exception as e:
            print(f"    Completion {i+1}/{len(old_completions)} ✗ Error: {e}")
            raise

    # Update data with new completions
    data["completions"] = new_completions

    # Determine output path
    save_path = output_path if output_path else filepath

    if dry_run:
        print(f"  [DRY RUN] Would save migrated file to: {save_path}")
        return

    # Backup original file (only if overwriting)
    if backup and not output_path:
        backup_path = filepath.with_suffix('.json.bak')
        shutil.copy2(filepath, backup_path)
        print(f"  Created backup: {backup_path}")

    # Save migrated file
    with open(save_path, 'w') as f:
        json.dump(data, f)

    print(f"  ✓ Migration complete - saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate rollout JSON files from old format to new format"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Rollout JSON files to migrate"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ByteDance-Seed/UI-TARS-1.5-7B",
        help="Model name for tokenizer/processor (default: Qwen/Qwen2.5-VL-7B-Instruct)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files (.bak)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually modifying files"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for migrated file (only works with single input file)"
    )

    args = parser.parse_args()

    # Validate --output usage
    if args.output and len(args.files) > 1:
        parser.error("--output can only be used with a single input file")

    print(f"Loading processor from {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    print("✓ Processor loaded")

    # Process each file
    success_count = 0
    error_count = 0

    for filepath in args.files:
        if not filepath.exists():
            print(f"\n✗ File not found: {filepath}")
            error_count += 1
            continue

        try:
            migrate_rollout_file(
                filepath,
                processor,
                backup=not args.no_backup,
                dry_run=args.dry_run,
                output_path=args.output
            )
            success_count += 1
        except Exception as e:
            print(f"\n✗ Error processing {filepath}: {e}")
            error_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Migration complete!")
    print(f"  Successfully processed: {success_count}")
    print(f"  Errors: {error_count}")
    print("=" * 60)


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("🪲 Waiting for debugger to attach on port 5678...")
    # debugpy.wait_for_client()
    main()
