import json
from pathlib import Path
from cua import Action, ActionType, State
from io import BytesIO
import base64
import httpx
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import List, Dict
from simple_data_entry import SimpleDataEntryTask


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


class UITARSRollout:
    def __init__(
        self, 
        task: SimpleDataEntryTask, 
        model_host: str, 
        model_name: str,
        httpx_client: httpx.AsyncClient, 
        max_images_in_context: int = 10,
        max_completion_tokens: int = 200,
        temperature: float = 1.0
    ):
        self._messages = [{
            "role": "user", "content": [{
                "type": "text",
                "text": UI_TARS_PROMPT.format(user_instruction=task.get_prompt()),
             }]
        }]
        self._completions: List[Completion] = []
        self._task = task
        self._progress = None
        self._model_host = model_host
        self._model_name = model_name
        self._client = httpx_client
        self._max_images_in_context = max_images_in_context
        self._max_completion_tokens = max_completion_tokens
        self._temperature = temperature

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, progress: Dict):
        self._progress = progress

    async def predict_next_action(self, new_state: State) -> Action | None:
        # Add new state to messages list
        self._messages.append({
            "role": "user", "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(new_state.screenshot)}"}
            }]
        })

        # Get messages to predict next action
        context = self._get_completion_context()
        response = await self._request_completion(
            model=self._model_name,
            messages=context,
            temperature=self._temperature,
            max_tokens=self._max_completion_tokens,
            skip_special_tokens=False,
            logprobs=1,
        )

        if "choices" not in response or len(response["choices"]) == 0:
            raise ValueError("No output from model")

        completion_message = response["choices"][0]["message"]
        logprobs = response["choices"][0]["logprobs"]["content"]

        completion = Completion(
            context=context,
            completion_message=completion_message,
            logprobs=logprobs
        )

        self._messages.append(completion_message)
        self._completions.append(completion)

        # Handle response message
        assert isinstance(completion_message["content"], str), "completion_message['content'] must be a string"
        thought, reflection, action_str = parse_response_string(completion_message["content"])
        if action_str in ("finished()", "call_user()"):
            return None

        return await parse_action(action_str)

    def _get_completion_context(self):
        messages = []
        messages.append(self._messages[0]) # Add prompt message

        # Add as much as possible of the message context, but cap at _max_images_in_context
        image_messages = [i for i, m in enumerate(self._messages) if "content" in m and isinstance(m["content"], list) and any(c.get("type") == "image_url" for c in m["content"])]
        if len(image_messages) <= self._max_images_in_context:
            # Include all context
            messages += self._messages[1:]
        else:
            # Cap to include at most _max_images_in_context
            cap_idx = image_messages[-self._max_images_in_context]
            messages += self._messages[cap_idx:]
        return messages

    async def _request_completion(self, **kwargs):
        for attempt in range(3):
            try:
                resp = await self._client.post(
                    url=f"http://{self._model_host}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=kwargs,
                )
                resp.raise_for_status()
                return resp.json()
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as e:
                if attempt < 2:  # Not the last attempt
                    logging.warning(f"Error predicting next action: {str(e)} (attempt {attempt + 1}/3)")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    logging.error(f"Request failed after 3 attempts: {str(e)}")
                    raise

    def save(self, filepath: str | Path):
        rollout_json = {
            "task": self._task.get_state(),
            "messages": self._messages,
            "completions": [
                {
                    "context": [self._messages.index(m) for m in completion.context],
                    "completion": self._messages.index(completion.completion_message),
                    "logprobs": completion.logprobs
                }
                for completion in self._completions
            ],
            "progress": self._progress
        }
        with open(filepath, 'w') as f:
            json.dump(rollout_json, f)



@dataclass
class Completion:
    context: List[int]
    completion_message: int
    logprobs: Dict


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

    else:
        return None

    
def encode_image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")