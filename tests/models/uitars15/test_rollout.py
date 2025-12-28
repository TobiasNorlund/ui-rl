import json
import base64
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from io import BytesIO

import pytest
from PIL import Image

from ui_rl.models.uitars15.rollout import (
    UITARS15_Rollout,
    Completion,
    parse_response_string,
    parse_action,
    encode_image_to_base64,
)
from ui_rl.cua import Action, ActionType, State
from ui_rl.task import TaskSpec


class MockTaskSpec(TaskSpec):
    """Mock implementation of TaskSpec for testing"""

    def __init__(self, instruction: str = "Test task instruction"):
        self.instruction = instruction

    def get_task_instruction(self) -> str:
        return self.instruction

    def create_session(self, runtime):
        return "mock_session_id"


@pytest.fixture
def mock_httpx_client():
    """Fixture that returns a mock httpx client"""
    client = Mock()
    return client


@pytest.fixture
def mock_task_spec():
    """Fixture that returns a mock task spec"""
    return MockTaskSpec()


@pytest.fixture
def sample_image():
    """Fixture that creates a simple test image"""
    img = Image.new('RGB', (100, 100), color='red')
    return img


class TestParseResponseString:
    """Tests for the parse_response_string function"""

    def test_parse_thought_and_action(self):
        response = "Thought: I need to click the button\nAction: click(start_box='<|box_start|>(100,200)<|box_end|>')"
        thought, reflection, action_str = parse_response_string(response)

        assert thought == "I need to click the button"
        assert reflection is None
        assert action_str == "click(start_box='<|box_start|>(100,200)<|box_end|>')"

    def test_parse_reflection_action_summary_and_action(self):
        response = "Reflection: This didn't work\nAction_Summary: Try clicking elsewhere\nAction: click(start_box='<|box_start|>(50,50)<|box_end|>')"
        thought, reflection, action_str = parse_response_string(response)

        assert thought == "Try clicking elsewhere"
        assert reflection == "This didn't work"
        assert action_str == "click(start_box='<|box_start|>(50,50)<|box_end|>')"

    def test_parse_action_summary_and_action(self):
        response = "Action_Summary: Click the submit button\nAction: click(start_box='<|box_start|>(300,400)<|box_end|>')"
        thought, reflection, action_str = parse_response_string(response)

        assert thought == "Click the submit button"
        assert reflection is None
        assert action_str == "click(start_box='<|box_start|>(300,400)<|box_end|>')"

    def test_parse_action_only(self):
        response = "click(start_box='<|box_start|>(100,200)<|box_end|>')"
        thought, reflection, action_str = parse_response_string(response)

        assert thought is None
        assert reflection is None
        assert action_str == "click(start_box='<|box_start|>(100,200)<|box_end|>')"

    def test_parse_finished_action(self):
        response = "Thought: Task is complete\nAction: finished()"
        thought, reflection, action_str = parse_response_string(response)

        assert thought == "Task is complete"
        assert reflection is None
        assert action_str == "finished()"


class TestParseAction:
    """Tests for the parse_action function"""

    def test_parse_click(self):
        action_str = "click(start_box='<|box_start|>(100,200)<|box_end|>')"
        action = parse_action(action_str)

        assert action is not None
        assert action.action_type == ActionType.LeftClick
        assert action.x == 100
        assert action.y == 200

    def test_parse_left_double(self):
        action_str = "left_double(start_box='<|box_start|>(150,250)<|box_end|>')"
        action = parse_action(action_str)

        assert action is not None
        assert action.action_type == ActionType.DoubleClick
        assert action.x == 150
        assert action.y == 250

    def test_parse_right_single(self):
        action_str = "right_single(start_box='<|box_start|>(300,400)<|box_end|>')"
        action = parse_action(action_str)

        assert action is not None
        assert action.action_type == ActionType.RightClick
        assert action.x == 300
        assert action.y == 400

    def test_parse_hotkey(self):
        action_str = "hotkey(key='ctrl+c')"
        action = parse_action(action_str)

        assert action is not None
        assert action.action_type == ActionType.Keys
        assert action.keys == "ctrl+c"

    def test_parse_hotkey_with_spaces(self):
        action_str = "hotkey(key='ctrl + c')"
        action = parse_action(action_str)

        assert action is not None
        assert action.action_type == ActionType.Keys
        assert action.keys == "ctrl+c"  # Spaces should be replaced with +

    def test_parse_type(self):
        action_str = "type(content='Hello World')"
        action = parse_action(action_str)

        assert action is not None
        assert action.action_type == ActionType.Type
        assert action.text == "Hello World"

    def test_parse_type_empty(self):
        action_str = "type(content='')"
        action = parse_action(action_str)

        assert action is not None
        assert action.action_type == ActionType.Type
        assert action.text == ""

    def test_parse_scroll(self):
        action_str = "scroll(start_box='<|box_start|>(500,600)<|box_end|>', direction='down')"
        action = parse_action(action_str)

        assert action is not None
        assert action.action_type == ActionType.Scroll
        assert action.x == 500
        assert action.y == 600
        assert action.direction == "down"

    def test_parse_scroll_up(self):
        action_str = "scroll(start_box='<|box_start|>(100,200)<|box_end|>', direction='up')"
        action = parse_action(action_str)

        assert action is not None
        assert action.action_type == ActionType.Scroll
        assert action.direction == "up"

    @patch('time.sleep')
    def test_parse_wait(self, mock_sleep):
        action_str = "wait()"
        action = parse_action(action_str)

        assert action is not None
        assert action.action_type == ActionType.Screenshot
        mock_sleep.assert_called_once_with(1)

    def test_parse_invalid_action(self):
        action_str = "invalid_action()"
        action = parse_action(action_str)

        assert action is None

    def test_parse_click_invalid_format(self):
        action_str = "click(invalid_format)"

        with pytest.raises(ValueError, match="Couldn't parse action"):
            parse_action(action_str)

    def test_parse_scroll_missing_direction(self):
        action_str = "scroll(start_box='<|box_start|>(500,600)<|box_end|>')"

        with pytest.raises(ValueError, match="Couldn't parse all required arguments"):
            parse_action(action_str)


class TestEncodeImageToBase64:
    """Tests for the encode_image_to_base64 function"""

    def test_encode_image(self, sample_image):
        encoded = encode_image_to_base64(sample_image)

        # Check that it's a valid base64 string
        assert isinstance(encoded, str)
        assert len(encoded) > 0

        # Verify we can decode it back
        decoded_bytes = base64.b64decode(encoded)
        decoded_image = Image.open(BytesIO(decoded_bytes))

        assert decoded_image.size == sample_image.size
        assert decoded_image.mode == sample_image.mode


class TestUITARS15Rollout:
    """Tests for the UITARS15_Rollout class"""

    def test_progress_property(self, mock_task_spec, mock_httpx_client):
        rollout = UITARS15_Rollout(
            task_spec=mock_task_spec,
            model_host="localhost:8000",
            model_name="test-model",
            httpx_client=mock_httpx_client
        )

        assert rollout.progress is None

        progress_data = {"score": 0.8, "completed": True}
        rollout.progress = progress_data

        assert rollout.progress == progress_data

    def test_get_prompt_message_indices_all_context(self, mock_task_spec, mock_httpx_client):
        rollout = UITARS15_Rollout(
            task_spec=mock_task_spec,
            model_host="localhost:8000",
            model_name="test-model",
            httpx_client=mock_httpx_client,
            max_images_in_context=10
        )

        # Add some messages (image + assistant response pairs)
        for i in range(3):
            rollout._messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,test{i}"}}]
            })
            rollout._messages.append({
                "role": "assistant",
                "content": f"Response {i}"
            })

        indices = rollout._get_prompt_message_indices()

        # Should include prompt (0) and all other messages (1-6)
        assert indices == [0, 1, 2, 3, 4, 5, 6]

    def test_get_prompt_message_indices_capped_context(self, mock_task_spec, mock_httpx_client):
        rollout = UITARS15_Rollout(
            task_spec=mock_task_spec,
            model_host="localhost:8000",
            model_name="test-model",
            httpx_client=mock_httpx_client,
            max_images_in_context=2  # Cap at 2 images
        )

        # Add 5 image + response pairs (10 messages total)
        for i in range(5):
            rollout._messages.append({
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,test{i}"}}]
            })
            rollout._messages.append({
                "role": "assistant",
                "content": f"Response {i}"
            })

        indices = rollout._get_prompt_message_indices()

        # Should include prompt (0) and only last 2 images worth of context
        # Images are at indices: 1, 3, 5, 7, 9
        # Last 2 images are at 7 and 9, so we include from 7 onwards: [7, 8, 9, 10]
        assert indices == [0, 7, 8, 9, 10]

    def test_save(self, mock_task_spec, mock_httpx_client, sample_image):
        rollout = UITARS15_Rollout(
            task_spec=mock_task_spec,
            model_host="localhost:8000",
            model_name="test-model",
            httpx_client=mock_httpx_client
        )

        # Add a completion
        rollout._messages.append({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}]
        })
        rollout._messages.append({
            "role": "assistant",
            "content": "Test response"
        })

        completion = Completion(
            prompt_token_ids=[1, 2, 3],
            prompt_messages=[{"role": "user", "content": "test"}],
            prompt_message_indices=[0, 1],
            generated_token_ids=[4, 5, 6],
            generated_message={"role": "assistant", "content": "Test response"},
            generated_message_index=2
        )
        rollout._completions.append(completion)
        rollout.progress = {"score": 0.9}

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            rollout.save(temp_path)

            # Read back and verify
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)

            assert "task" in saved_data
            assert "messages" in saved_data
            assert "completions" in saved_data
            assert "progress" in saved_data

            assert saved_data["task"]["instruction"] == "Test task instruction"
            assert len(saved_data["messages"]) == 3
            assert len(saved_data["completions"]) == 1
            assert saved_data["progress"] == {"score": 0.9}

            # Verify completion format
            saved_completion = saved_data["completions"][0]
            assert saved_completion["prompt_token_ids"] == [1, 2, 3]
            assert saved_completion["prompt_messages"] == [0, 1]  # Should be indices
            assert saved_completion["generated_token_ids"] == [4, 5, 6]
            assert saved_completion["generated_message"] == 2  # Should be index
        finally:
            Path(temp_path).unlink()

    def test_predict_next_action_finished(self, mock_task_spec, mock_httpx_client, sample_image):
        mock_httpx_client.post.return_value = Mock(
            json=lambda: {
                "choices": [{
                    "message": {"role": "assistant", "content": "Thought: Task complete\nAction: finished()"},
                    "token_ids": [1, 2, 3]
                }],
                "prompt_token_ids": [10, 11, 12]
            },
            raise_for_status=lambda: None
        )

        rollout = UITARS15_Rollout(
            task_spec=mock_task_spec,
            model_host="localhost:8000",
            model_name="test-model",
            httpx_client=mock_httpx_client
        )

        state = State(screenshot=sample_image)
        action = rollout.predict_next_action(state)

        assert action is None
        assert len(rollout._messages) == 3  # prompt + image + response
        assert len(rollout._completions) == 1

    def test_predict_next_action_click(self, mock_task_spec, mock_httpx_client, sample_image):
        mock_httpx_client.post.return_value = Mock(
            json=lambda: {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Thought: Click the button\nAction: click(start_box='<|box_start|>(100,200)<|box_end|>')"
                    },
                    "token_ids": [1, 2, 3]
                }],
                "prompt_token_ids": [10, 11, 12]
            },
            raise_for_status=lambda: None
        )

        rollout = UITARS15_Rollout(
            task_spec=mock_task_spec,
            model_host="localhost:8000",
            model_name="test-model",
            httpx_client=mock_httpx_client
        )

        state = State(screenshot=sample_image)
        action = rollout.predict_next_action(state)

        assert action is not None
        assert action.action_type == ActionType.LeftClick
        assert action.x == 100
        assert action.y == 200
        assert len(rollout._completions) == 1

        # Verify the request was made with correct headers
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args.kwargs["headers"]["X-Routing-ID"] == rollout._rollout_id

    def test_request_completion_retry(self, mock_task_spec, mock_httpx_client):
        import httpx

        # Mock to fail twice, then succeed
        mock_httpx_client.post.side_effect = [
            httpx.HTTPError("Connection error"),
            httpx.HTTPError("Connection error"),
            Mock(
                json=lambda: {"choices": [{"message": {"role": "assistant", "content": "test"}}]},
                raise_for_status=lambda: None
            )
        ]

        rollout = UITARS15_Rollout(
            task_spec=mock_task_spec,
            model_host="localhost:8000",
            model_name="test-model",
            httpx_client=mock_httpx_client
        )

        result = rollout._request_completion(messages=[])

        assert result == {"choices": [{"message": {"role": "assistant", "content": "test"}}]}
        assert mock_httpx_client.post.call_count == 3

    def test_request_completion_all_retries_fail(self, mock_task_spec, mock_httpx_client):
        import httpx

        mock_httpx_client.post.side_effect = httpx.HTTPError("Connection error")

        rollout = UITARS15_Rollout(
            task_spec=mock_task_spec,
            model_host="localhost:8000",
            model_name="test-model",
            httpx_client=mock_httpx_client
        )

        with pytest.raises(httpx.HTTPError):
            rollout._request_completion(messages=[])

        assert mock_httpx_client.post.call_count == 3
