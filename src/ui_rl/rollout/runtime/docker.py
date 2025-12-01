from datetime import datetime
from typing import Optional
from dataclasses import asdict
from urllib.parse import quote
from PIL import Image
import logging
import io
import uuid
import asyncio
import httpx
import docker  # type: ignore
from docker.models.containers import Container  # type: ignore
from . import CUASessionRuntime
from ..cua import State, Action


logger = logging.getLogger(__name__)


class DockerSessionRuntime(CUASessionRuntime):
    """
    Launches sessions as Docker containers.
    Connects to containers via their internal Docker network IP address.
    """

    def __init__(
        self,
        port: int = 8000,
        session_timeout: int = 30,  # Timeout in seconds for session to come online
        httpx_client: Optional[httpx.AsyncClient] = None,
        docker_client: Optional[docker.DockerClient] = None,
        **container_kwargs
    ):
        """
        Args:
            port: Port used for communicating with the session server
            session_timeout: Timeout in seconds for session to come online
            httpx_client: Optional httpx client for making HTTP requests
            docker_client: Optional Docker client, defaults to docker.from_env()
            **container_kwargs: Additional kwargs to pass to docker.containers.run()
        """
        self._port = port
        self._docker_client = docker_client or docker.from_env()
        self._session_timeout = session_timeout
        self._httpx_client = httpx_client or httpx.AsyncClient(timeout=30.0)
        self._container_kwargs = container_kwargs
        self._containers: dict[str, Container] = {}

    def create_session(self, **kwargs) -> str:
        session_id = str(uuid.uuid4())[:8]
        container_name = f"session-{session_id}"

        # Run container in detached mode with --rm (auto-remove on stop)
        container = self._docker_client.containers.run(
            name=container_name,
            detach=True,
            remove=True,
            **kwargs,
            **self._container_kwargs
        )

        self._containers[session_id] = container
        return session_id

    def teardown_session(self, session_id: str):
        container = self._containers.get(session_id)
        if container:
            try:
                container.stop()
            except Exception as e:
                logger.warning(f"({session_id}) Error stopping container: {e}")
            finally:
                del self._containers[session_id]

    async def session_ready(self, session_id: str):
        container = self._containers.get(session_id)
        if not container:
            raise RuntimeError(f"Session {session_id} not found")

        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < self._session_timeout:
            try:
                # Check if container is still running
                await asyncio.to_thread(container.reload)
                if container.status != "running":
                    raise RuntimeError(f"Session {session_id} container stopped unexpectedly")

                # Get container's internal IP address
                container_ip = self._get_container_ip(container)

                # Try to connect to the container's HTTP server via internal IP
                resp = await self._httpx_client.get(
                    f"http://{container_ip}:{self._port}/",
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
        container = self._containers.get(session_id)
        if not container:
            raise RuntimeError(f"Session {session_id} not found")

        # Get the container's internal IP
        container_ip = self._get_container_ip(container)

        qs = "&".join(f"{k}={quote(str(v))}" for k, v in asdict(action).items() if v is not None)
        url = f"http://{container_ip}:{self._port}/act?{qs}"

        # Retry up to 3 times on 5xx errors
        for attempt in range(3):
            try:
                resp = await self._httpx_client.get(url)
                # Check for 5xx server errors
                if resp.status_code >= 500:
                    if attempt < 2:  # Not the last attempt
                        logger.warning(f"({session_id}) Error acting: HTTP {resp.status_code} {str(resp.content)} (attempt {attempt + 1}/3)")
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
        else:
            raise RuntimeError() # Won't reach here

    async def get_session_progress(self, session_id: str) -> dict:
        container = self._containers.get(session_id)
        if not container:
            raise RuntimeError(f"Session {session_id} not found")

        # Get the container's internal IP
        #await asyncio.to_thread(container.reload)
        container_ip = self._get_container_ip(container)

        url = f"http://{container_ip}:{self._port}/progress"

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
        else:
            raise RuntimeError()  # won't reach here

    def _get_container_ip(self, container: Container) -> str:
        """
        Extract the container's internal IP address from the default bridge network.
        """
        # Refresh container info to get latest network settings
        #container.reload()

        # Get network settings
        networks = container.attrs.get('NetworkSettings', {}).get('Networks', {})

        if not networks:
            raise RuntimeError(f"No network configuration found for container {container.name}")

        # Get the first network's IP address (usually 'bridge' for default Docker network)
        for network_name, network_config in networks.items():
            ip_address = network_config.get('IPAddress')
            if ip_address:
                return ip_address

        raise RuntimeError(f"No IP address found for container {container.name}")
