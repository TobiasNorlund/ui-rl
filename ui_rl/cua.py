import enum
import httpx
import asyncio
import io
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable
from PIL import Image
from functools import partial
from kubernetes import client, config
from urllib.parse import quote
from simple_data_entry import SimpleDataEntryTask


def load_kube_config():
    # Load config and k10s client globally
    global core_v1
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


async def run_cua_rollout(
    rollout,
    task: SimpleDataEntryTask,
    cluster_host: str,
    max_steps: int = 10,
    session_timeout: int = 60 * 2,  # Timeout in seconds for session to come online
    httpx_client: httpx.AsyncClient = None
):
    """
    Launches a session pod, awaits it ready, and executes a Computer Use agent in it.
    """
    session_id = str(uuid.uuid4())[:8]
    log = partial(_log, session_id=session_id)
    pod_name = f"session-{session_id}"

    # Create pod
    core_v1.create_namespaced_pod(
        namespace="default",
        body=task.get_pod_manifest(pod_name, session_id)
    )

    try:
        log("Starting...")
        await _await_ready(cluster_host, session_id, session_timeout, httpx_client)
        log("Ready")

        # Start with getting the init state (e.g. take screenshot)
        action = Action(ActionType.Screenshot)
        state = await _act(cluster_host, session_id, action, httpx_client)

        for step_num in range(max_steps):
            log(f"Predicting action {step_num+1}")
            action = await rollout.predict_next_action(state)
            if action is None:
                break
                
            log(f"Taking action: {action}")
            state = await _act(cluster_host, session_id, action, httpx_client)

        # Get final rollout progress
        progress = await _get_progress(cluster_host, session_id, httpx_client)
        rollout.progress = progress

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

async def _await_ready(cluster_host: str, session_id: str, session_timeout: int, session: httpx.AsyncClient):
    start_time = datetime.now()
    while (datetime.now() - start_time).seconds < session_timeout:
        try:
            resp = await session.get(
                f"http://{cluster_host}/proxy/{session_id}/",
            )
            if resp.status_code == 200:
                break
            else:
                #content = resp.content
                #_log(f"{resp}: {content}", session_id=session_id)
                await asyncio.sleep(2)
        except (httpx.HTTPError, asyncio.TimeoutError):
            await asyncio.sleep(2)
            continue
        except asyncio.CancelledError:
            raise
    else:
        raise RuntimeError(f"Session {session_id} never came up")


async def _act(cluster_host: str, session_id: str, action: Action, session: httpx.AsyncClient) -> State:
    """
    Acts in the session via proxy server running on k10s cluster
    """
    log = partial(_log, session_id=session_id)
    qs = "&".join(f"{k}={quote(str(v))}" for k, v in asdict(action).items() if v is not None)
    url = f"http://{cluster_host}/proxy/{session_id}/act?{qs}"

    # Retry up to 3 times on 5xx errors
    for attempt in range(3):
        try:
            resp = await session.get(url)
            # Check for 5xx server errors
            if resp.status_code >= 500:
                if attempt < 2:  # Not the last attempt
                    log(f"Error acting: HTTP {resp.status_code} {str(resp.content)} (attempt {attempt + 1}/3)", level=logging.WARNING)
                    continue
                else:
                    resp.raise_for_status()  # Raise exception on last attempt

            # Check for other HTTP errors
            resp.raise_for_status()

            # Parse response bytes as PIL Image
            try:
                content = resp.content
                image = Image.open(io.BytesIO(content))
                return State(image)
            except Exception as e:
                log(f"Failed to parse response as image: {e}", level=logging.ERROR)
                raise ValueError(f"Invalid image response: {e}")

        except asyncio.CancelledError:
            raise

        except httpx.HTTPError as e:
            if attempt < 2:  # Not the last attempt
                log(f"Error acting: {str(e)} (attempt {attempt + 1}/3)", level=logging.WARNING)
                await asyncio.sleep(1)
                #breakpoint()
                continue
            else:
                log(f"Act failed after 3 attempts: {str(e)}", level=logging.ERROR)
                raise


async def _get_progress(cluster_host, session_id, session: httpx.AsyncClient) -> Dict:
    """
    Get progress information from the session pod via proxy server
    """
    log = partial(_log, session_id=session_id)
    url = f"http://{cluster_host}/proxy/{session_id}/progress"

    # Retry up to 3 times on 5xx errors
    for attempt in range(3):
        try:
            resp = await session.get(url)
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
                log(f"Failed to parse progress response as JSON: {e}", level=logging.ERROR)
                raise ValueError(f"Invalid progress response: {e}")

        except httpx.HTTPError as e:
            if attempt < 2:  # Not the last attempt
                log(f"Error getting progress: {str(e)} (attempt {attempt + 1}/3)", level=logging.WARNING)
                await asyncio.sleep(1)
                continue
            else:
                log(f"Failed getting progress after 3 attempts: {str(e)}", level=logging.ERROR)
                raise


def _log(msg: str, session_id: str, level=logging.INFO):
    logging.log(level=level, msg=f"CUASession {session_id}: {msg}")
