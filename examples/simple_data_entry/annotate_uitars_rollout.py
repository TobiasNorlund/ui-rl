import streamlit as st
import json
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from litellm import completion

# Set appropriate app width 
st.html("""
    <style>
        .stMainBlockContainer {
            max-width: 70rem;
        }
    </style>
    """
)


UI_TARS_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

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
Your task is to submit data from a spreadsheet (seen on the left) into a form (seen on the right). Specifically, the following rows (as numbered in the left margin) from the spreadsheet are to be submitted: {rows}.
Note: You may need to scroll to make the row visible in the sheet.
The form has to be submitted separately for each row. When the form has been submitted, return to the form to submit the next row. 
Submit a row by selecting each cell individually, copy its content by sending keys "ctrl+c", select the target form text input and paste using "ctrl+v".
Finally, click "Skicka" to submit the form, and continue with the next row. Only finish when all rows have been successfully submitted"""


SUGGEST_RESPONSE_PROMPT = """The following is an instruction to a GUI agent, to be performed on a computer using the GUI:

--------
{system_prompt}
--------

The agent has already performed the following actions:
{prev_actions}

Now, the the agent is provided the following screenshot and the natural next step is to perform the following action: {next_action}

Write the corresponding output, according to the format specified above, for this action. Write only the output and nothing else.
"""


def decode_base64_image(base64_str: str):
    if base64_str.startswith("data:image"):
        header, base64_data = base64_str.split(",", 1)
    else:
        base64_data = base64_str
    img_bytes = base64.b64decode(base64_data)
    buffer = BytesIO(img_bytes)
    return Image.open(buffer)


def get_event_display_string(event: dict):
    event_type = {
        "left_click": "LeftClick",
        "right_click": "RightClick",
        "double_click": "DoubleClick",
        "triple_click": "TripleClick",
        "scroll": "Scroll",
        "type": "Type",
        "keys": "Keys",
    }
    args = [f"{k}={v}" for k, v in event.items() if k not in ['action_type', 'timestamp'] and v is not None]
    return f"{event_type[event['action_type']]}({', '.join(args)})" 


def image_viewer(event_index: int, screenshot_suggestion: int):
    index_var = f"screenshot_{event_index}_index"
    if index_var not in st.session_state:
        st.session_state[index_var] = screenshot_suggestion

    def next_image():
        st.session_state[index_var] = (st.session_state[index_var] + 1)
    def prev_image():
        st.session_state[index_var] = (st.session_state[index_var] - 1)

    col1, col2, col3 = st.columns([1, 12, 1])
    with col1:
        st.button("⬅", on_click=prev_image, key=f"prev_{event_index}")
    with col2:
        current_image = screenshots[st.session_state[index_var]]["image"].copy()
        if event_index < len(recording["events"]):
            event = recording["events"][event_index]
            if 'x' in event and 'y' in event and event['x'] is not None and event['y'] is not None:
                draw = ImageDraw.Draw(current_image)
                radius = 12
                x = int(event['x'])
                y = int(event['y'])
                left_up_point = (x - radius, y - radius)
                right_down_point = (x + radius, y + radius)
                draw.ellipse([left_up_point, right_down_point], outline="red", width=4)
        st.image(current_image, caption=f"Screenshot {st.session_state[index_var]}", width="stretch")
    with col3:
        st.button("➡", on_click=next_image, key=f"next_{event_index}")


def suggest_screenshot_for_event(event_index: int):
    # Suggest the first screenshot that comes after the previous event to event_index
    if event_index == 0:
        return 0
    
    prev_event = recording["events"][event_index-1]
    for i, screenshot in enumerate(recording["screenshots"]):
        if screenshot["timestamp"] > prev_event["timestamp"] + 0.5:
            return i
    else:
        return 0


def format_uitars_action(event):
    """Format a ProcessedAction event into UITARS action format"""
    action_type = event["action_type"]
    x = event.get("x")
    y = event.get("y")
    
    match action_type:
        case "left_click":
            return f"click(start_box='<|box_start|>({x},{y})<|box_end|>')"
        
        case "double_click":
            return f"left_double(start_box='<|box_start|>({x},{y})<|box_end|>')"
        
        case "triple_click":
            # UITARS doesn't have triple_click, treat as double click
            return f"left_double(start_box='<|box_start|>({x},{y})<|box_end|>')"
        
        case "right_click":
            return f"right_single(start_box='<|box_start|>({x},{y})<|box_end|>')"
        
        case "type":
            text = event.get("text", "")
            # Escape single quotes in the text
            text_escaped = text.replace("'", "\\'")
            return f"type(content='{text_escaped}')"
        
        case "keys":
            keys = event.get("keys", [])
            if isinstance(keys, list):
                # Format keys list as 'key1+key2' (e.g., 'ctrl+c')
                key_str = "+".join(str(k).lower() for k in keys)
            else:
                key_str = str(keys).lower()
            return f"hotkey(key='{key_str}')"
        
        case "scroll":
            dx = event.get("dx", 0)
            dy = event.get("dy", 0)
            
            # Determine scroll direction
            if abs(dy) > abs(dx):
                direction = "down" if dy < 0 else "up"
            else:
                direction = "right" if dx < 0 else "left"
            
            if x is not None and y is not None:
                return f"scroll(start_box='<|box_start|>({x},{y})<|box_end|>', direction='{direction}')"
            else:
                return f"scroll(start_box='<|box_start|>(0,0)<|box_end|>', direction='{direction}')"
        
        case _:
            return f"# Unknown action type: {action_type}"


def suggest_response(event_index: int):

    prev_actions = ""
    for i in range(event_index):
        prev_actions += f"{i+1}:\n{st.session_state[f'response_{i}']}\n\n"
    
    if event_index < len(recording["events"]):
        next_action = format_uitars_action(recording["events"][event_index])
    else:
        next_action = "finished()"

    resp = completion(
        model="gemini/gemini-2.5-flash",
        messages=[{
            "role": "user", 
            "content": [
                {
                    "type": "text", 
                    "text": SUGGEST_RESPONSE_PROMPT.format(
                        system_prompt=UI_TARS_SYSTEM_PROMPT.format(rows=rows),
                        prev_actions=prev_actions,
                        next_action=next_action
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + recording["screenshots"][st.session_state[f"screenshot_{event_index}_index"]]["screenshot_base64"]
                    }
                }
            ]
        }]
    )
    return resp["choices"][0]["message"]["content"]


def get_rollout() -> dict:
    """Build a UITARS rollout-like payload ready for download."""
    # 1. Base prompt message
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": UI_TARS_SYSTEM_PROMPT.format(rows=rows)
        }]
    }]

    # 2. Per-event screenshot + assistant response
    num_events = len(recording["events"]) + 1  # include finalizing action
    image_message_indices = []
    completions = []
    for i in range(num_events):
        screenshot_idx = st.session_state.get(f"screenshot_{i}_index", 0)
        screenshot_b64 = recording["screenshots"][screenshot_idx]["screenshot_base64"]

        messages.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
            }]
        })
        image_message_indices.append(len(messages)-1)

        response_text = st.session_state.get(f"response_{i}", "")
        messages.append({
            "role": "assistant",
            "content": response_text
        })

        completions.append(
            {
                "context": [0] + list(range(image_message_indices[max(0, len(image_message_indices)-int(st.session_state.max_images))], image_message_indices[-1] + 1)),
                "completion": len(messages) - 1
            }
        )

    rollout = {
        "task": {"rows": [int(r) for r in rows.split(", ")]},
        "messages": messages,
        "completions": completions,
        "progress": progress_data,
    }

    return rollout


# --- App ---

uploaded_file = st.file_uploader("Select a recording file (*.json)")
if uploaded_file is not None:
    # Store recording in session state so modifications persist across reruns
    if "recording" not in st.session_state or st.session_state.get("uploaded_file_name") != uploaded_file.name:
        recording = json.loads(uploaded_file.getvalue().decode("utf-8"))
        st.session_state["recording"] = recording
        st.session_state["uploaded_file_name"] = uploaded_file.name
    else:
        recording = st.session_state["recording"]

    screenshots = [
        {"timestamp": screenshot["timestamp"], "image": decode_base64_image(screenshot["screenshot_base64"])}
        for screenshot in recording["screenshots"]
    ]

    st.title("Simple Data Entry Annotation Tool")

    rows = st.text_input("Rows", placeholder="2, 3, 4")

    st.subheader("System Prompt")
    st.code(UI_TARS_SYSTEM_PROMPT.format(rows=rows))
    st.divider()

    def make_suggest_callback(event_idx):
        def suggest_callback():
            response = suggest_response(event_idx)
            st.session_state[f"response_{event_idx}"] = response
        return suggest_callback

    def make_delete_callback(event_idx):
        def delete_callback():
            # Remove the event from the recording
            del recording["events"][event_idx]

            # Clean up session state for this and all subsequent events
            # We need to shift all the state keys down by one
            for i in range(event_idx, len(recording["events"]) + 1):
                # Shift response_{i+1} to response_{i}
                if f"response_{i+1}" in st.session_state:
                    st.session_state[f"response_{i}"] = st.session_state[f"response_{i+1}"]
                    del st.session_state[f"response_{i+1}"]
                else:
                    # Clear the current index if there's no next one
                    if f"response_{i}" in st.session_state:
                        del st.session_state[f"response_{i}"]

                # Shift screenshot_{i+1}_index to screenshot_{i}_index
                if f"screenshot_{i+1}_index" in st.session_state:
                    st.session_state[f"screenshot_{i}_index"] = st.session_state[f"screenshot_{i+1}_index"]
                    del st.session_state[f"screenshot_{i+1}_index"]
                else:
                    # Clear the current index if there's no next one
                    if f"screenshot_{i}_index" in st.session_state:
                        del st.session_state[f"screenshot_{i}_index"]
        return delete_callback

    for i, event in enumerate(recording["events"]):
        col1, col2 = st.columns([7, 1])
        with col1:
            st.markdown(f"##### {i+1}. {get_event_display_string(event)}")
        with col2:
            st.button("🗑️", key=f"delete_{i}", on_click=make_delete_callback(i))
        col1, col2 = st.columns([7, 1])
        with col1:
            st.text_area("Response", placeholder="Thought: I will ...", key=f"response_{i}")
        with col2:
            st.space("medium")
            st.button("Suggest", icon="🪄", on_click=make_suggest_callback(i), key=f"suggest_{i}")
        image_viewer(i, suggest_screenshot_for_event(i))
        st.divider()

    # Finalizing action
    i += 1
    st.markdown(f"##### {i+1}. Finalizing action")
    col1, col2 = st.columns([7, 1])
    with col1:
        st.text_area("Response", placeholder="Thought: I will ...", key=f"response_{i}")
    with col2:
        st.space("medium")
        st.button("Suggest", icon="🪄", on_click=make_suggest_callback(i), key=f"suggest_{i}")
    image_viewer(i, len(screenshots)-1)
    st.divider()

    st.text("Progress")
    progress_data = {"submitted_row_indices": [int(r)-2 for r in rows.split(", ") if r.strip()], "num_incorrect_submissions": 0}
    st.json(progress_data)

    st.text_input("Max images in completion", value=10, key="max_images")
    st.divider()

    if st.button("Generate Rollout"):
        st.session_state["rollout_data"] = get_rollout()

    if "rollout_data" in st.session_state:
        json_data = json.dumps(st.session_state["rollout_data"], ensure_ascii=False)
        st.download_button("Download Rollout", data=json_data, file_name=uploaded_file.name)