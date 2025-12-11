from io import BytesIO
from pathlib import Path
import h5py
import pickle
from PIL import Image
import asyncio
import logging
import uuid
import re
from transformers import AutoProcessor
from typing import List, NamedTuple
from vllm import AsyncLLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

from ui_rl.cua import Action, ActionType, State
from ui_rl.task import TaskSpec


logger = logging.getLogger(__name__)


UI_TARS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x1,y1)<|box_end|>')
hotkey(key='') # Press a hotkey, e.g. 'ctrl+c' or 'alt+f4'
type(content='') #If you want to submit your input, use "\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{user_instruction}
"""


class UITARS15_Rollout:

    _processor = AutoProcessor.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")

    def __init__(
        self, 
        task_spec: TaskSpec, 
        async_engine: AsyncLLMEngine,
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
        max_images_in_context: int = 10,
    ):
        self._task_spec = task_spec
        self._progress: dict | None = None
        self._engine = async_engine
        self._lora_request = lora_request
        self._sampling_params = sampling_params
        self._max_images_in_context = max_images_in_context

        self._messages = [
            {"role": "user", "content": [
                {"type": "text", "text": UI_TARS_PROMPT.format(user_instruction=self._task_spec.get_task_instruction())}
            ]}
        ]
        self._completions: List[Completion] = []
        self._images = []

    @property
    def progress(self) -> dict | None:
        return self._progress

    @progress.setter
    def progress(self, progress: dict):
        self._progress = progress

    async def predict_next_action(self, new_state: State) -> Action | None:
        # Append a new user message for the new screenshot
        self._messages.append(
            {"role": "user", "content": [
                {"type": "image"}
            ]}
        )
        self._images.append(new_state.screenshot)

        # Get the prompt messages for the next completion
        prompt_messages, images = self._get_prompt_messages_and_images()

        # Tokenize
        prompt_token_ids = self._processor.apply_chat_template(
            prompt_messages, 
            tokenize=True, 
            add_generation_prompt=True
        )[0]

        res_gen = self._engine.generate(
            prompt={"prompt_token_ids": prompt_token_ids, "multi_modal_data": {"image": images}},
            sampling_params=self._sampling_params,
            request_id=str(uuid.uuid4()),
            lora_request=self._lora_request
        )

        # Await generation...
        async for request_output in res_gen:
            # Just pass until the final request_output
            pass

        assert len(request_output.outputs) == 1, "Should only get one final output from vLLM"

        if request_output.outputs[0].finish_reason != "stop":
            logger.warning(f"Generation did not finish normally: {request_output.outputs[0].finish_reason}")

        generated_text = request_output.outputs[0].text
        generated_token_ids = request_output.outputs[0].token_ids

        self._completions.append(Completion(
            prompt_token_ids=prompt_token_ids,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            images=images
        ))

        self._messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": request_output.outputs[0].text}
            ]
        })

        # Handle response message
        thought, reflection, action_str = parse_response_string(generated_text)
        if action_str in ("finished()", "call_user()"):
            return None

        return await parse_action(action_str)

    def _get_prompt_messages_and_images(self):
        messages = []
        messages.append(self._messages[0]) # Add prompt message

        # Add as much as possible of the message context, but cap at _max_images_in_context
        image_messages = [i for i, m in enumerate(self._messages) if "content" in m and isinstance(m["content"], list) and any(c.get("type") == "image" for c in m["content"])]
        if len(image_messages) <= self._max_images_in_context:
            # Include all context
            messages += self._messages[1:]
            images = self._images
        else:
            # Cap to include at most _max_images_in_context
            cap_idx = image_messages[-self._max_images_in_context]
            messages += self._messages[cap_idx:]
            images = self._images[-10:]
        return messages, images

    def save(self, filepath: str | Path):
        with h5py.File(filepath, 'w') as f:
            for i, c in enumerate(self._completions):
                f[f"completions/{i}/prompt_token_ids"] = c.prompt_token_ids
                f[f"completions/{i}/generated_text"] = c.generated_text
                f[f"completions/{i}/generated_token_ids"] = c.generated_token_ids
                f[f"completions/{i}/images"] = [self._images.index(img) for img in c.images]
            f["images"] = [png_encode(img) for img in self._images]
            f["task_spec"] = pickle.dumps(self._task_spec.as_dict())
            f["progress"] = pickle.dumps(self._progress)


class Completion(NamedTuple):
    prompt_token_ids: list[int]
    generated_text: str
    generated_token_ids: list[int]
    images: list[Image]


def parse_response_string(response_string: str):
    thought = None
    reflection = None
    action_str = None

    if response_string.startswith("Thought:"):
        thought_match = re.search(r"Thought: ([\s\S]+?)(?=\s*Action:|$)", response_string)
        if thought_match:
            thought = thought_match.group(1).strip()

    elif response_string.startswith("Reflection:"):
        reflection_match = re.search(
            r"Reflection: ([\s\S]+?)Action_Summary: ([\s\S]+?)(?=\s*Action:|$)", response_string
        )
        if reflection_match:
            reflection = reflection_match.group(1).strip()
            thought = reflection_match.group(2).strip()

    elif response_string.startswith("Action_Summary:"):
        summary_match = re.search(r"Action_Summary: (.+?)(?=\s*Action:|$)", response_string)
        if summary_match:
            thought = summary_match.group(1).strip()

    if "Action:" not in response_string:
        action_str = response_string.strip()
    else:
        action_parts = response_string.split("Action:")
        action_str = action_parts[-1].strip()
    
    return thought, reflection, action_str


async def parse_action(action_str: str) -> Action | None:
    action_str = action_str.strip()
    if action_str.startswith("click("):
        match = re.search(r"click\(start_box='<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>'\)", action_str)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return Action(action_type=ActionType.LeftClick, x=x, y=y)
        else:
            raise ValueError(f"Couldn't parse action: {action_str}")
        
    elif action_str.startswith("left_double("):
        match = re.search(r"left_double\(start_box='<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>'\)", action_str)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return Action(action_type=ActionType.DoubleClick, x=x, y=y)
        else:
            raise ValueError(f"Couldn't parse action: {action_str}")
        
    elif action_str.startswith("right_single("):
        match = re.search(r"right_single\(start_box='<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>'\)", action_str)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return Action(action_type=ActionType.RightClick, x=x, y=y)
        else:
            raise ValueError(f"Couldn't parse action: {action_str}")
        
    elif action_str.startswith("hotkey("):
        match = re.search(r"hotkey\(key='([^']+)'\)", action_str)
        if match:
            key = match.group(1)
            return Action(action_type=ActionType.Keys, keys=key.replace(" ", "+"))
        else:
            raise ValueError(f"Couldn't parse action: {action_str}")
        
    elif action_str.startswith("type("):
        match = re.search(r"type\(content='([^']*)'\)", action_str)
        if match:
            content = match.group(1)
            return Action(action_type=ActionType.Type, text=content)
            # TODO: If the content ends with \n, we assume it's a submit action
        else:
            raise ValueError(f"Couldn't parse action: {action_str}")
        
    elif action_str.startswith("scroll("):
        # 1. Regex to extract start_box coordinates (x and y)
        box_regex = r"start_box='<\|box_start\|>\((?P<x>\d+),(?P<y>\d+)\)<\|box_end\|>'"
        box_match = re.search(box_regex, action_str)

        # 2. Regex to extract direction
        direction_regex = r"direction='(?P<direction>[^']+)'"
        direction_match = re.search(direction_regex, action_str)

        # Check if both required parts were found
        if box_match and direction_match:
            try:
                # Extract and convert coordinates
                x = int(box_match.group('x'))
                y = int(box_match.group('y'))
                
                # Extract direction
                direction = direction_match.group('direction')
                
                return Action(action_type=ActionType.Scroll, direction=direction, x=x, y=y)
            except ValueError as e:
                # Catch potential errors during integer conversion
                raise ValueError(f"Error converting coordinates in action: {action_str}. Details: {e}")
        else:
            raise ValueError(f"Couldn't parse all required arguments in action: {action_str}")
        
    elif action_str.startswith("wait()"):
        await asyncio.sleep(1)
        return Action(action_type=ActionType.Screenshot)
    else:
        return None


def png_encode(image: Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()