import enum
import requests
import io
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Callable
from PIL import Image, ImageDraw
from functools import partial
from kubernetes import client, config
from pathlib import Path
from urllib.parse import quote
from simple_data_entry import SimpleDataEntryTask


# Load config and k10s client globally
config.load_kube_config()
core_v1 = client.CoreV1Api()


@dataclass
class State:
    screenshot: Image


class ActionType(enum.StrEnum):
    Screenshot = "screenshot"
    MouseMove = "mouse_move"
    LeftClick = "left_click"
    RightClick = "right_click"
    DoubleClick = "double_click"
    TripleClick = "triple_click"
    Type = "type"
    Keys = "keys"

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


@dataclass
class Action:
    action_type: ActionType
    x: int | None = None
    y: int | None = None
    text: str | None = None
    keys: str | None = None


@dataclass
class Rollout:
    task: str
    states: List[State] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    response_messages: List[str] = field(default_factory=list)
    reward: float = 0.0
    progress: Dict = field(default_factory=dict)


def run_cua_session(
    task: SimpleDataEntryTask,
    predict_next_action: Callable,
    cluster_host: str,
    max_steps: int = 10,
    session_timeout: int = 60 * 5,  # Timeout in seconds for session to come online
    log_dir: Path | None = None
) -> Rollout:
    """
    Launches a session pod, awaits it ready, and executes a Computer Use agent in it.
    """
    session_id = str(uuid.uuid4())[:8]
    log = partial(_log, session_id=session_id)
    pod_name = f"session-{session_id}"
    rollout = Rollout(task=task.get_prompt())

    # Create pod
    core_v1.create_namespaced_pod(
        namespace="default",
        body=task.get_pod_manifest(pod_name, session_id)
    )

    try:
        log("Starting...")
        _await_ready(cluster_host, session_id, session_timeout)
        log("Ready")

        # Create log directory if specified
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        # Start with screenshot
        action = Action(ActionType.Screenshot)
        state = _act(cluster_host, session_id, action)
        rollout.actions.append(action)
        rollout.states.append(state)

        # Save initial screenshot
        if log_dir is not None:
            state.screenshot.save(log_dir / f"step_{len(rollout.states)-1:03d}.png")

        # Enter loop
        step_num = len(rollout.states)
        while action is not None and len(rollout.actions) < max_steps:
            screenshot = rollout.states[-1].screenshot.copy()
            action, message = predict_next_action(rollout)
            if action is not None:
                log(f"{action}")
                draw = ImageDraw.Draw(screenshot)
                draw.text((10, 10), str(action), fill="red")
                if action.x is not None and action.y is not None:
                    radius = 10
                    draw.ellipse(
                        [(action.x - radius, action.y - radius),
                         (action.x + radius, action.y + radius)],
                        outline="red",
                        width=3
                    )

                state = _act(cluster_host, session_id, action)
                rollout.actions.append(action)
                rollout.states.append(state)
                rollout.response_messages.append(message)

            # Save screenshot
            if log_dir is not None:
                screenshot.save(log_dir / f"step_{step_num:03d}.png")

            step_num += 1

        # Compute reward
        progress = _get_progress(cluster_host, session_id)
        rollout.progress = progress
        rollout.reward = task.get_reward(progress)

        log("Finished successfully")
        return rollout
    except Exception as e:
        import traceback
        log(f"ERROR in {pod_name}: {traceback.format_exc()}", level=logging.ERROR)
    finally:
        # Delete the pod
        core_v1.delete_namespaced_pod(
            name=pod_name,
            namespace='default'
        )

def _await_ready(cluster_host: str, session_id: str, session_timeout: int):
    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < session_timeout:
        try:
            resp = requests.get(f"http://{cluster_host}/proxy/{session_id}", timeout=5)
            if resp.status_code == 200:
                break
        except requests.exceptions.Timeout:
            continue


def _act(cluster_host: str, session_id: str, action: Action) -> State:
    """
    Acts in the session via proxy server running on k10s cluster
    """
    log = partial(_log, session_id=session_id)
    qs = "&".join(f"{k}={quote(str(v))}" for k, v in asdict(action).items() if v is not None)
    url = f"http://{cluster_host}/proxy/{session_id}/act?{qs}"
    
    # Retry up to 3 times on 5xx errors
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            
            # Check for 5xx server errors
            if resp.status_code >= 500:
                if attempt < 2:  # Not the last attempt
                    log(f"Server error {resp.status_code}, retrying... (attempt {attempt + 1}/3)")
                    continue
                else:
                    resp.raise_for_status()  # Raise exception on last attempt
            
            # Check for other HTTP errors
            resp.raise_for_status()
            
            # Parse response bytes as PIL Image
            try:
                image = Image.open(io.BytesIO(resp.content))
                return State(image)
            except Exception as e:
                log(f"Failed to parse response as image: {e}", level=logging.ERROR)
                raise ValueError(f"Invalid image response: {e}")
                
        except requests.exceptions.RequestException as e:
            if attempt < 2:  # Not the last attempt
                log(f"Request failed, retrying... (attempt {attempt + 1}/3): {e}")
                continue
            else:
                log(f"Request failed after 3 attempts: {e}", level=logging.ERROR)
                raise


def _get_progress(cluster_host, session_id) -> Dict:
    """
    Get progress information from the session pod via proxy server
    """
    log = partial(_log, session_id=session_id)
    url = f"http://{cluster_host}/proxy/{session_id}/progress"
    
    # Retry up to 3 times on 5xx errors
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            
            # Check for 5xx server errors
            if resp.status_code >= 500:
                if attempt < 2:  # Not the last attempt
                    log(f"Server error {resp.status_code}, retrying... (attempt {attempt + 1}/3)")
                    continue
                else:
                    resp.raise_for_status()  # Raise exception on last attempt
            
            # Check for other HTTP errors
            resp.raise_for_status()
            
            # Parse response as JSON
            try:
                progress = resp.json()
                return progress
            except Exception as e:
                log(f"Failed to parse progress response as JSON: {e}", logging.ERROR)
                raise ValueError(f"Invalid progress response: {e}")
                
        except requests.exceptions.RequestException as e:
            if attempt < 2:  # Not the last attempt
                log(f"Request failed, retrying... (attempt {attempt + 1}/3): {e}")
                continue
            else:
                log(f"Request failed after 3 attempts: {e}", logging.ERROR)
                raise

def _log(msg: str, session_id: str, level=logging.INFO):
    logging.log(level=level, msg=f"CUASession {session_id}: {msg}")
