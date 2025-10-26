from cua import Rollout, Action, ActionType
from io import BytesIO
import base64
import requests
import re
import logging
from time import sleep


FULL_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

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


def predict_next_action(rollout: Rollout, model_host: str) -> Action | None:
    messages = [
        {"role": "user", "content": FULL_PROMPT.format(user_instruction=rollout.task)},
    ]

    # TODO: Condense long rollouts
    max_imgs = 10
    # Take last max_imgs-1, cause we're adding the latest image below the loop
    for state, message in zip(rollout.states[-max_imgs:-1], rollout.response_messages[-max_imgs+1:]):
        messages.append({
            "role": "user", "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(state.screenshot)}"}
            }]
        })
        messages.append({
            "role": "assistant", "content": [{
                "type": "text",
                "text": message
            }]
        })

    # Add last screenshot
    messages.append({
        "role": "user", "content": [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(rollout.states[-1].screenshot)}"}
        }]
    })

    response = create_response(
        host=model_host,
        #model="ui-tars",
        model="ByteDance-Seed/UI-TARS-1.5-7B",
        messages=messages,
        temperature=0.1,
        max_tokens=200
    )

    if "choices" not in response or len(response["choices"]) == 0:
        raise ValueError("No output from model")

    message = response["choices"][0]["message"]

    # Handle response message
    assert isinstance(message["content"], str), "message['content'] must be a string"
    thought, reflection, action_str = parse_response_string(message["content"])
    logging.info(f"Thought: {thought}")
    logging.info(f"Action: {action_str}")

    if action_str in ("finished()", "call_user()"):
        return None, message["content"]

    return parse_action(action_str), message["content"]


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


def parse_action(action_str: str) -> Action | None:
    action_str = action_str.strip()
    if action_str.startswith("click("):
        match = re.search(r"click\(start_box='\((\d+),(\d+)\)'\)", action_str)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return Action(action_type=ActionType.LeftClick, x=x, y=y)
        else:
            raise ValueError(f"Couldn't parse action: {action_str}")
        
    elif action_str.startswith("left_double("):
        match = re.search(r"left_double\(start_box='\((\d+),(\d+)\)'\)", action_str)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            return Action(action_type=ActionType.DoubleClick, x=x, y=y)
        else:
            raise ValueError(f"Couldn't parse action: {action_str}")
        
    elif action_str.startswith("right_single("):
        match = re.search(r"right_single\(start_box='\((\d+),(\d+)\)'\)", action_str)
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
        match = re.search(r"scroll\(start_box='\((\d+),(\d+)\)', direction='([^']+)'\)", action_str)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            direction = match.group(3)
            raise NotImplementedError()
        else:
            raise ValueError(f"Couldn't parse action: {action_str}")
        
    elif action_str.startswith("wait()"):
        sleep(1)

    else:
        return None


def create_response(host, **kwargs):
    resp = requests.post(
        url=f"http://{host}/v1/chat/completions",
        headers={"Content-Type": "application/json"}, 
        json=kwargs
    )
    resp.raise_for_status()
    return resp.json()

    
def encode_image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")