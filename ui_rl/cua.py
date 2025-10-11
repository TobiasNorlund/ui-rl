import multiprocessing as mp
import enum
from operator import rshift
import requests
import logging
import io
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Callable, Dict
from collections import namedtuple
from PIL import Image
from functools import partial
from kubernetes import client, config


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
    states: List[State] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    reward: float = 0.0


# TODO: Perhaps better to feed KV cache + latest state?
ActionPredictionInput = namedtuple("ActionPredictionInput", ["rollout", "result_queue"])


def run_cua_session(
    pod_manifest_fn: Callable,
    cua_inference_queue: mp.Queue,
    reward_fn: Callable,
    cluster_host: str,
    max_steps: int = 10,
    session_timeout: int = 60 * 5  # Timeout in seconds for session to come online
):
    """
    Launches a session pod, awaits it ready, and executes a Computer Use agent in it.
    """
    session_id = str(uuid.uuid4())[:8]
    log = partial(_log, session_id=session_id)
    pod_name = f"session-{session_id}"
    rollout = Rollout()

    # Create pod
    core_v1.create_namespaced_pod(
        namespace="default",
        body=pod_manifest_fn(pod_name, session_id)
    )

    try:
        log("Starting...")
        _await_ready(cluster_host, session_id, session_timeout)
        log("Ready")

        # Start with screenshot
        action = Action(ActionType.Screenshot)
        state = _act(cluster_host, session_id, action)
        rollout.actions.append(action)
        rollout.states.append(state)

        # Enter loop
        prediction_result_queue = mp.Manager().Queue()
        cua_inference_queue.put(
            ActionPredictionInput(rollout, prediction_result_queue)
        )
        # TODO: Get updated KV cache?
        while action := prediction_result_queue.get():
            log(f"{action}")
            state = _act(cluster_host, session_id, action)
            rollout.actions.append(action)
            rollout.states.append(state)

            # Queue next action prediction
            if len(rollout.actions) < max_steps:
                cua_inference_queue.put(
                    ActionPredictionInput(rollout, prediction_result_queue)
                )
            else:
                break

        # Compute reward
        progress = _get_progress(cluster_host, session_id)
        rollout.reward = reward_fn(progress)

        log("Finished successfully")
    except Exception as e:
        log(f"Failed to delete pod {pod_name}: {e}", logging.ERROR)
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
            resp = requests.get(f"http://{cluster_host}:8000/proxy/{session_id}", timeout=5)
            if resp.status_code == 200:
                break
        except requests.exceptions.Timeout:
            continue


def _act(cluster_host: str, session_id: str, action: Action) -> State:
    """
    Acts in the session via proxy server running on k10s cluster
    """
    log = partial(_log, session_id=session_id)
    qs = "&".join(f"{k}={v}" for k, v in asdict(action).items() if v is not None)
    url = f"http://{cluster_host}:8000/proxy/{session_id}/act?{qs}"
    
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
                log(f"Failed to parse response as image: {e}", logging.ERROR)
                raise ValueError(f"Invalid image response: {e}")
                
        except requests.exceptions.RequestException as e:
            if attempt < 2:  # Not the last attempt
                log(f"Request failed, retrying... (attempt {attempt + 1}/3): {e}")
                continue
            else:
                log(f"Request failed after 3 attempts: {e}", logging.ERROR)
                raise


def _get_progress(cluster_host, session_id) -> Dict:
    """
    Get progress information from the session pod via proxy server
    """
    log = partial(_log, session_id=session_id)
    url = f"http://{cluster_host}:8000/proxy/{session_id}/progress"
    
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
