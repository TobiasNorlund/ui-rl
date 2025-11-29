from datetime import datetime
from accelerate.tracking import LoggerType
from kubernetes import client, config
from typing import Callable
from dataclasses import asdict
from urllib.parse import quote
from PIL import Image
import logging
import io
import uuid
import asyncio
import httpx
from . import CUASessionRuntime
from ..cua import State, Action


logger = logging.getLogger(__name__)


class KubernetesSessionRuntime(CUASessionRuntime):
    f"""
    Launches sessions as pods on a Kubernetes cluster.
    Assumes a running proxy server deployed on the cluster at `host` that forwards requests to individual session pods via:

       http://{{host}}/proxy/{{session_id}}/...
    """

    def __init__(
        self, 
        manifest_fn: Callable, 
        host: str, 
        session_timeout: int = 60 * 3,  # Timeout in seconds for session to come online
        httpx_client: httpx.AsyncClient = None
    ):
        config.load_kube_config()
        self._core_v1 = client.CoreV1Api()
        self._manifest_fn = manifest_fn
        self._host = host
        self._session_timeout = session_timeout
        self._httpx_client = httpx_client

    async def create_session(self) -> str:
        session_id = str(uuid.uuid4())[:8]
        pod_name = f"session-{session_id}"
        self._core_v1.create_namespaced_pod(
            namespace="default",
            body=self._manifest_fn(pod_name, session_id)
        )
        return session_id

    async def teardown_session(self, session_id: str):
        self._core_v1.delete_namespaced_pod(
            name=f"session-{session_id}",
            namespace='default'
        )

    async def session_ready(self, session_id: str):
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < self._session_timeout:
            try:
                resp = await self._httpx_client.get(
                    f"http://{self._host}/proxy/{session_id}/",
                )
                if resp.status_code == 200:
                    break
                else:
                    await asyncio.sleep(2)
            except (httpx.HTTPError, asyncio.TimeoutError):
                await asyncio.sleep(2)
                continue
            except asyncio.CancelledError:
                raise
        else:
            raise RuntimeError(f"Session {session_id} never came up")

    async def session_act(self, session_id: str, action: Action) -> State:
        qs = "&".join(f"{k}={quote(str(v))}" for k, v in asdict(action).items() if v is not None)
        url = f"http://{self._host}/proxy/{session_id}/act?{qs}"

        # Retry up to 3 times on 5xx errors
        for attempt in range(3):
            try:
                resp = await self._httpx_client.get(url)
                # Check for 5xx server errors
                if resp.status_code >= 500:
                    if attempt < 2:  # Not the last attempt
                        LoggerType.warning(f"({session_id}) Error acting: HTTP {resp.status_code} {str(resp.content)} (attempt {attempt + 1}/3)")
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
                    logger.error(f"({session_id}) Failed to parse response as image: {e}")
                    raise ValueError(f"Invalid image response: {e}")

            except asyncio.CancelledError:
                raise

            except httpx.HTTPError as e:
                if attempt < 2:  # Not the last attempt
                    logger.warning(f"({session_id}) Error acting: {str(e)} (attempt {attempt + 1}/3)")
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"({session_id}) Act failed after 3 attempts: {str(e)}")
                    raise

    async def get_session_progress(self, session_id: str) -> dict:
        url = f"http://{self._host}/proxy/{session_id}/progress"

        # Retry up to 3 times on 5xx errors
        for attempt in range(3):
            try:
                resp = await self._httpx_client.get(url)
                # Check for 5xx server errors
                if resp.status_code >= 500:
                    if attempt < 2:  # Not the last attempt
                        logger.warning(f"({session_id}) Server error {resp.status_code}, retrying... (attempt {attempt + 1}/3)")
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
                    logger.error(f"({session_id}) Failed to parse progress response as JSON: {e}")
                    raise ValueError(f"Invalid progress response: {e}")

            except httpx.HTTPError as e:
                if attempt < 2:  # Not the last attempt
                    logger.warning(f"({session_id}) Error getting progress: {str(e)} (attempt {attempt + 1}/3)")
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"({session_id}) Failed getting progress after 3 attempts: {str(e)}")
                    raise
