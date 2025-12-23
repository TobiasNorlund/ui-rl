import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import random
from itertools import islice
import pytest
import torch
from PIL import Image
import io
import base64

from transformers import AutoProcessor

from ui_rl.models.uitars15.dataset import (
    UITARS15_RolloutDataset,
    UITARS15_ThoughtAugmentedRolloutDataset,
)


@pytest.fixture
def mock_processor():
    """Fixture that returns a mock processor"""
    processor = Mock()

    # Mock image processor
    image_processor = Mock()
    image_processor.return_value = {
        "pixel_values": torch.randn(1, 3, 224, 224),
        "image_grid_thw": torch.tensor([[1, 224, 224]])
    }
    processor.image_processor = image_processor

    # Mock apply_chat_template
    processor.apply_chat_template = Mock(return_value={
        "input_ids": torch.LongTensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 5),
        "pixel_values": torch.randn(1, 3, 224, 224),
        "image_grid_thw": torch.tensor([[1, 224, 224]])
    })

    return processor


@pytest.fixture
def sample_image_base64():
    """Create a sample base64-encoded image"""
    img = Image.new('RGB', (100, 100), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@pytest.fixture
def sample_rollout_json(sample_image_base64):
    """Create a sample rollout JSON structure"""
    return {
        "task": {
            "instruction": "Test task",
            "type": "test"
        },
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "System prompt"}]
            },
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": sample_image_base64}}]
            },
            {
                "role": "assistant",
                "content": "Action 1"
            },
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": sample_image_base64}}]
            },
            {
                "role": "assistant",
                "content": "Action 2"
            }
        ],
        "completions": [
            {
                "prompt_token_ids": [1, 2, 3, 4, 5],
                "generated_token_ids": [6, 7, 8],
                "prompt_messages": [0, 1],
                "generated_message": 2
            },
            {
                "prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "generated_token_ids": [11, 12, 13],
                "prompt_messages": [0, 1, 2, 3],
                "generated_message": 4
            }
        ],
        "progress": {"score": 0.8}
    }


@pytest.fixture
def sample_augmented_rollout_json(sample_image_base64):
    """Create a sample augmented rollout JSON with alternative completions"""
    return {
        "task": {
            "instruction": "Test task",
            "type": "test"
        },
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "System prompt"}]
            },
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": sample_image_base64}}]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": ["Action 1 variant A", "Action 1 variant B", "Action 1 variant C"]
                    }
                ]
            },
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": sample_image_base64}}]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": ["Action 2 variant A", "Action 2 variant B"]
                    }
                ]
            }
        ],
        "completions": [
            {
                "prompt_messages": [0, 1],
                "generated_message": 2
            },
            {
                "prompt_messages": [0, 1, 2, 3],
                "generated_message": 4
            }
        ],
        "progress": {"score": 0.8}
    }


class TestUITARS15RolloutDataset:
    """Tests for UITARS15_RolloutDataset"""

    def test_initialization(self, mock_processor, sample_rollout_json, tmp_path):
        # Create temporary rollout file
        rollout_file = tmp_path / "test_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(mock_processor, str(rollout_file))

        assert dataset._processor == mock_processor
        assert len(dataset._sequences) > 0

    def test_len(self, mock_processor, sample_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(mock_processor, str(rollout_file))

        # Should have sequences after merging
        assert len(dataset) == len(dataset._sequences)
        assert len(dataset) > 0

    def test_task_spec(self, mock_processor, sample_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_augmented_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(
            mock_processor,
            str(rollout_file)
        )

        assert dataset.task_spec == {
            "instruction": "Test task",
            "type": "test"
        }

    def test_getitem_structure(self, mock_processor, sample_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(mock_processor, str(rollout_file))
        item = dataset[0]

        # Check required keys
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert "pixel_values" in item
        assert "image_grid_thw" in item

        # Check tensor types and shapes
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

        # Check shapes match
        assert item["input_ids"].shape == item["attention_mask"].shape
        assert item["input_ids"].shape == item["labels"].shape

    def test_getitem_labels_masking(self, mock_processor, sample_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(mock_processor, str(rollout_file))
        item = dataset[0]

        # Labels should have -100 for non-completion tokens
        labels = item["labels"]
        assert (labels == -100).any()  # Some tokens should be masked

        # Some tokens should not be masked (completion tokens)
        assert (labels != -100).any()

    def test_load_rollout(self, mock_processor, sample_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(mock_processor, str(rollout_file))
        rollout = dataset._load_rollout(str(rollout_file))

        # Check rollout structure
        assert rollout.task_spec == sample_rollout_json["task"]
        assert rollout.progress == sample_rollout_json["progress"]
        assert isinstance(rollout.sequences, list)
        assert len(rollout.sequences) > 0

        # Verify that token IDs in completion spans match the generated_token_ids from the JSON
        for seq in rollout.sequences:
            for completion_span in seq.completions:
                # Extract the tokens in the completion span
                completion_tokens = seq.token_ids[completion_span.start:completion_span.end].tolist()

                # Find the corresponding completion in the original JSON
                # The completion tokens should match one of the generated_token_ids
                found_match = False
                for completion in sample_rollout_json["completions"]:
                    if completion["generated_token_ids"] == completion_tokens:
                        found_match = True
                        break

                assert found_match, f"Completion tokens {completion_tokens} not found in rollout JSON"

    def test_sequence_merging(self, mock_processor, sample_rollout_json, tmp_path):
        """Test that sequences with common prefixes are merged"""
        rollout_file = tmp_path / "test_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(mock_processor, str(rollout_file))
        rollout = dataset._load_rollout(str(rollout_file))

        # The shorter sequence should be merged into the longer one
        # We should have fewer sequences than completions
        assert len(rollout.sequences) <= len(sample_rollout_json["completions"])

        # At least one sequence should have multiple completions
        has_multiple_completions = any(len(seq.completions) > 1 for seq in rollout.sequences)
        assert has_multiple_completions

    def test_token_sequence_structure(self, mock_processor, sample_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(mock_processor, str(rollout_file))
        seq = dataset._sequences[0]

        # Check TokenSequence structure
        assert isinstance(seq.token_ids, torch.Tensor)
        assert isinstance(seq.completions, list)
        assert isinstance(seq.base64_images, list)

        # Check that completions are valid spans
        for completion in seq.completions:
            assert completion.start >= 0
            assert completion.end <= len(seq.token_ids)
            assert completion.start < completion.end

    def test_base64_image_extraction(self, mock_processor, sample_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(mock_processor, str(rollout_file))

        seq = dataset._sequences[0]

        # Should have extracted base64 images
        assert len(seq.base64_images) > 0

        # Images should be valid base64 strings
        for img_b64 in seq.base64_images:
            assert "data:image" in img_b64
            assert "base64" in img_b64


class TestUITARS15ThoughtAugmentedRolloutDataset:
    """Tests for UITARS15_ThoughtAugmentedRolloutDataset"""

    def test_task_spec(self, mock_processor, sample_augmented_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_augmented_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_augmented_rollout_json, f)

        dataset = UITARS15_ThoughtAugmentedRolloutDataset(
            mock_processor,
            str(rollout_file),
            random_seed=42
        )

        assert dataset.task_spec == {
            "instruction": "Test task",
            "type": "test"
        }

    def test_iter_structure(self, sample_augmented_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_augmented_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_augmented_rollout_json, f)

        processor = AutoProcessor.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
        dataset = UITARS15_ThoughtAugmentedRolloutDataset(
            processor,
            str(rollout_file),
            random_seed=42
        )

        # Get one item from iterator
        item = next(iter(dataset))

        # Check required keys
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert "pixel_values" in item
        assert "image_grid_thw" in item

        # Check tensor types
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

    def test_find_message_spans(self):
        """Test the _find_message_spans class method"""
        # Create a mock token sequence with message delimiters
        message_start_id = 151644
        message_end_id = 151645
        user_role_id = 151643
        assistant_role_id = 77091

        input_ids = [
            message_start_id, user_role_id, 100, 101, 102, message_end_id,  # First message
            message_start_id, assistant_role_id, 200, 201, message_end_id,  # Second message
            message_start_id, user_role_id, 300, 301, 302, 303, message_end_id  # Third message
        ]

        spans = UITARS15_ThoughtAugmentedRolloutDataset._find_message_spans(input_ids)

        assert len(spans) == 3

        # Check first span
        assert spans[0].start == 0
        assert spans[0].end == 5
        assert spans[0].role_id == user_role_id

        # Check second span
        assert spans[1].start == 6
        assert spans[1].end == 10
        assert spans[1].role_id == assistant_role_id

        # Check third span
        assert spans[2].start == 11
        assert spans[2].end == 17
        assert spans[2].role_id == user_role_id

    def test_find_message_spans_empty(self):
        """Test _find_message_spans with empty input"""
        spans = UITARS15_ThoughtAugmentedRolloutDataset._find_message_spans([])
        assert len(spans) == 0

    def test_find_message_spans_no_delimiters(self):
        """Test _find_message_spans with no message delimiters"""
        input_ids = [1, 2, 3, 4, 5]
        spans = UITARS15_ThoughtAugmentedRolloutDataset._find_message_spans(input_ids)
        assert len(spans) == 0

    def test_consistency(self, sample_augmented_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_augmented_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_augmented_rollout_json, f)

        processor = AutoProcessor.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
        dataset = UITARS15_ThoughtAugmentedRolloutDataset(
            processor,
            str(rollout_file),
            random_seed=42
        )

        for ex1, ex2 in islice(zip(dataset, dataset), 3):
            assert set(ex1.keys()) == set(ex2.keys())
            assert all(torch.all(ex1[k] == ex2[k]) for k in ex1.keys())

    def test_multiworker_consistency(self, sample_augmented_rollout_json, tmp_path):
        rollout_file = tmp_path / "test_augmented_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_augmented_rollout_json, f)

        processor = AutoProcessor.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
        dataset = UITARS15_ThoughtAugmentedRolloutDataset(
            processor,
            str(rollout_file),
            random_seed=42
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
        for ex1, ex2 in islice(zip(dataloader, dataloader), 3):
            assert set(ex1.keys()) == set(ex2.keys())
            assert all(torch.all(ex1[k] == ex2[k]) for k in ex1.keys())

class TestDatasetIntegration:
    """Integration tests for the datasets"""

    def test_rollout_dataset_full_pipeline(self, mock_processor, sample_rollout_json, tmp_path):
        """Test the full pipeline from file to batch"""
        rollout_file = tmp_path / "test_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(mock_processor, str(rollout_file))

        # Should be able to iterate through all items
        for i in range(len(dataset)):
            item = dataset[i]
            assert item is not None
            assert "input_ids" in item
            assert "labels" in item

    def test_multiple_sequences_have_consistent_structure(self, mock_processor, sample_rollout_json, tmp_path):
        """Test that all sequences have consistent structure"""
        rollout_file = tmp_path / "test_rollout.json"
        with open(rollout_file, 'w') as f:
            json.dump(sample_rollout_json, f)

        dataset = UITARS15_RolloutDataset(mock_processor, str(rollout_file))

        if len(dataset) > 1:
            item1 = dataset[0]
            item2 = dataset[1]

            # Should have same keys
            assert set(item1.keys()) == set(item2.keys())
