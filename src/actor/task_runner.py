"""TaskRunner for executing UI tasks and collecting trajectories using Kubernetes pods."""

from typing import Dict, Any, Optional, Callable
from pathlib import Path
import logging
from datetime import datetime
import time
import uuid
import numpy as np
import torch
import torch.nn as nn
import requests
from PIL import Image
from io import BytesIO
from kubernetes import client, config as k8s_config

from ..data_utils.trajectory import Trajectory
from ..data_utils.actions_decoder import ActionsDecoder

logger = logging.getLogger(__name__)

# Load Kubernetes config globally
try:
    k8s_config.load_kube_config()
    core_v1 = client.CoreV1Api()
except Exception as e:
    logger.warning(f"Failed to load Kubernetes config: {e}. Pod creation will not work.")
    core_v1 = None


class TaskRunner:
    """
    Actor component: Runs episodes in Kubernetes pods and collects trajectories.

    This implementation creates dedicated Kubernetes pods for each session:
    - Creates pod via Kubernetes API
    - Waits for pod to be ready
    - Communicates via proxy server: http://{cluster_host}:8000/proxy/{session_id}/*
    - GET /proxy/{session_id}/act: Perform action and get resulting screenshot
    - GET /proxy/{session_id}/progress: Get reward/progress feedback
    - Deletes pod when episode completes

    Design decisions:
    1. One pod per episode (clean isolation)
    2. Has reference to model for inference (shared with Trainer)
    3. Collects full trajectories before sending to training
    4. Handles screenshot preprocessing
    5. Uses ActionsDecoder to parse VLM outputs into API actions

    Responsibilities:
    - Create Kubernetes pods for sessions
    - Wait for pod readiness
    - Run inference with VLM to get actions
    - Execute actions via proxy server
    - Collect rewards from progress endpoint
    - Delete pods when episode completes
    - Build trajectory and return to caller
    """

    def __init__(
        self,
        cluster_host: str,
        pod_manifest_fn: Callable[[str, str], Dict],
        model: nn.Module,
        task_prompt: str,
        namespace: str = "default",
        max_steps_per_episode: int = 50,
        screenshot_size: tuple = (224, 224),
        data_dir: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        action_format: str = "json",
        session_timeout: int = 300,
    ):
        """
        Args:
            cluster_host: Kubernetes cluster proxy server host/IP
            pod_manifest_fn: Function that generates pod manifest given (pod_name, session_id)
            model: VLM for action inference (shared reference with Trainer)
            task_prompt: Task description to send to VLM
            namespace: Kubernetes namespace for pods
            max_steps_per_episode: Maximum steps before episode terminates
            screenshot_size: Resize screenshots to this size (H, W)
            data_dir: Optional directory to save raw trajectories
            device: Device for model inference
            action_format: Format for action decoding ("json", "text", "coordinates", "natural")
            session_timeout: Timeout in seconds for pod to come online
        """
        self.cluster_host = cluster_host
        self.pod_manifest_fn = pod_manifest_fn
        self.namespace = namespace
        self.session_timeout = session_timeout
        self.model = model
        self.model.eval()  # Important: set to eval mode for inference
        self.task_prompt = task_prompt
        self.max_steps_per_episode = max_steps_per_episode
        self.screenshot_size = screenshot_size
        self.data_dir = Path(data_dir) if data_dir else None
        self.device = device

        # Action decoder for parsing VLM outputs
        self.actions_decoder = ActionsDecoder(default_format=action_format)

        # State for current episode
        self.current_session_id: Optional[str] = None
        self.current_pod_name: Optional[str] = None
        self.step_count = 0

        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TaskRunner initialized with cluster: {cluster_host}, namespace: {namespace}")

    def _preprocess_screenshot(self, screenshot: Image.Image) -> Image.Image:
        """
        Preprocess screenshot for VLM input.

        Note: Returns PIL Image directly to avoid unnecessary conversions.
        VLMWrapper expects PIL Images as input.

        Args:
            screenshot: PIL Image screenshot

        Returns:
            Preprocessed screenshot as PIL Image (currently no-op, can add resize/crop later)
        """
        # Optional: Resize to standard size if needed
        # img = screenshot.resize(self.screenshot_size, Image.Resampling.BILINEAR)

        # Return PIL Image directly (no numpy conversion)
        return screenshot

    def _get_action(self, screenshot: Image.Image, prompt: str) -> Dict[str, Any]:
        """
        Run VLM inference to get action.

        Args:
            screenshot: PIL Image screenshot
            prompt: Task prompt

        Returns:
            Action dict, e.g. {"action_type": "left_click", "x": 100, "y": 200}
            Generated text, e.g. "left_click(100, 200)"
        """
        with torch.no_grad():  # No gradients during inference
            # VLMWrapper.predict_action accepts PIL Image directly
            generated_text = self.model.predict_action(
                images=screenshot,
                prompt=prompt
            )

            # Decode the generated text into action dict
            action = self.actions_decoder.decode(generated_text)

            # Validate action
            if not self.actions_decoder.validate_action(action):
                logger.warning(f"Invalid action generated: {action}")
                # Fallback to screenshot action
                action = {"action_type": "screenshot"}

        return action, generated_text

    def _execute_action(self, action: Dict[str, Any]) -> tuple:
        """
        Send action to UI environment via proxy server and get next state.

        Args:
            action: Action dict to execute

        Returns:
            (next_screenshot, reward, done, info)
        """
        try:
            # Convert action to API parameters
            params = self.actions_decoder.action_to_api_params(action)

            # Execute action via proxy server with retries
            # The request blocks until the pod completes the action
            url = f"http://{self.cluster_host}:8000/proxy/{self.current_session_id}/act"
            screenshot = self._request_with_retries(url, params, is_image=True)

            # Get progress/reward from separate endpoint
            progress_url = f"http://{self.cluster_host}:8000/proxy/{self.current_session_id}/progress"
            progress_data = self._request_with_retries(progress_url, {}, is_image=False)

            # Calculate reward based on progress
            reward = self._calculate_reward(progress_data)

            # Check if episode should end
            #done = self._check_done(progress_data)
            done = False

            return screenshot, reward, done, progress_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error executing action: {e}")
            # Return dummy values and mark as done due to error
            dummy_screenshot = Image.new('RGB', self.screenshot_size, color='black')
            return dummy_screenshot, -1.0, True, {"error": str(e)}

    def _request_with_retries(self, url: str, params: Dict, is_image: bool, max_retries: int = 3):
        """
        Make HTTP request with retries for 5xx errors.

        Args:
            url: URL to request
            params: Query parameters
            is_image: If True, parse response as PIL Image; if False, parse as JSON
            max_retries: Maximum number of retry attempts

        Returns:
            PIL Image or Dict depending on is_image parameter
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)

                # Check for 5xx server errors
                if response.status_code >= 500:
                    if attempt < max_retries - 1:
                        logger.warning(f"Server error {response.status_code}, retrying... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        response.raise_for_status()

                # Check for other HTTP errors
                response.raise_for_status()

                # Parse response
                if is_image:
                    return Image.open(BytesIO(response.content))
                else:
                    return response.json()

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Request failed, retrying... (attempt {attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    raise

    def _calculate_reward(self, progress_data: Dict[str, Any]) -> float:
        """
        Calculate reward from progress data.

        Args:
            progress_data: Progress data from /progress endpoint

        Returns:
            Reward value for this step
        """
        # Task-specific reward calculation
        # For simple_data_entry: reward for correct submissions
        if "num_correct_submissions" in progress_data:
            # Get change in correct submissions since last step
            prev_correct = getattr(self, '_prev_correct_submissions', 0)
            current_correct = progress_data["num_correct_submissions"]
            reward = float(current_correct - prev_correct)
            self._prev_correct_submissions = current_correct
            return reward

        # Default: small negative reward per step to encourage efficiency
        return -0.01

    def _check_done(self, progress_data: Dict[str, Any]) -> bool:
        """
        Check if episode should terminate based on progress.

        Args:
            progress_data: Progress data from /progress endpoint

        Returns:
            True if episode should end
        """
        # Task-specific termination conditions
        # For simple_data_entry: Could end after N correct submissions
        if "num_correct_submissions" in progress_data:
            # End after 5 successful submissions
            if progress_data["num_correct_submissions"] >= 5:
                return True

        # Check for errors
        if "error" in progress_data:
            return True

        return False

    def _create_session(self) -> tuple:
        """
        Create a new Kubernetes pod for the session.

        Returns:
            Tuple of (session_id, pod_name)
        """
        if core_v1 is None:
            raise RuntimeError("Kubernetes client not initialized. Cannot create pods.")

        try:
            # Generate unique session ID and pod name
            session_id = str(uuid.uuid4())[:8]
            pod_name = f"session-{session_id}"

            # Create pod using provided manifest function
            pod_manifest = self.pod_manifest_fn(pod_name, session_id)

            # Create pod via Kubernetes API
            core_v1.create_namespaced_pod(
                namespace=self.namespace,
                body=pod_manifest
            )

            logger.info(f"Created pod {pod_name} with session ID {session_id}")

            # Wait for pod to be ready
            self._await_ready(session_id)
            logger.info(f"Pod {pod_name} is ready")

            return session_id, pod_name

        except Exception as e:
            logger.error(f"Error creating pod: {e}")
            raise

    def _await_ready(self, session_id: str):
        """
        Wait for pod to be ready by checking proxy endpoint.

        Args:
            session_id: Session ID to check

        Raises:
            TimeoutError: If pod doesn't become ready within session_timeout
        """
        start_time = time.time()
        url = f"http://{self.cluster_host}:8000/proxy/{session_id}"

        while (time.time() - start_time) < self.session_timeout:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return
            except requests.exceptions.Timeout:
                continue
            except requests.exceptions.RequestException:
                # Pod not ready yet
                time.sleep(1)
                continue

            time.sleep(1)

        raise TimeoutError(f"Pod with session {session_id} did not become ready within {self.session_timeout}s")

    def _close_session(self, session_id: str, pod_name: str):
        """
        Delete the Kubernetes pod for this session.

        Args:
            session_id: Session ID
            pod_name: Pod name to delete
        """
        if core_v1 is None:
            logger.warning("Kubernetes client not initialized. Cannot delete pod.")
            return

        try:
            core_v1.delete_namespaced_pod(
                name=pod_name,
                namespace=self.namespace
            )
            logger.info(f"Deleted pod {pod_name} (session {session_id})")

        except Exception as e:
            logger.error(f"Error deleting pod {pod_name}: {e}")

    def _get_screenshot(self, session_id: str) -> Image.Image:
        """
        Get initial screenshot from session via proxy server.

        This is used to get the first screenshot after pod creation.
        Subsequent screenshots are obtained via _execute_action.

        Args:
            session_id: Session ID

        Returns:
            PIL Image of screenshot
        """
        try:
            # Get screenshot via screenshot action
            url = f"http://{self.cluster_host}:8000/proxy/{session_id}/act"
            params = {"action_type": "screenshot"}
            screenshot = self._request_with_retries(url, params, is_image=True)
            return screenshot

        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting screenshot: {e}")
            # Return blank image on error
            return Image.new('RGB', self.screenshot_size, color='black')

    def run_episode(self) -> Trajectory:
        """
        Run a single episode and return the complete trajectory.

        Creates a Kubernetes pod, runs the episode, and deletes the pod.

        Returns:
            Completed trajectory with PIL Images in observations
        """
        # Create new session (pod)
        self.current_session_id, self.current_pod_name = self._create_session()

        # Get initial screenshot (PIL Image)
        screenshot = self._get_screenshot(self.current_session_id)
        screenshot = self._preprocess_screenshot(screenshot)

        # Initialize trajectory
        observations = []
        generated_texts = []
        rewards = []
        prompts = []

        done = False
        step = 0

        # Reset progress tracking
        self._prev_correct_submissions = 0

        logger.info(f"Starting episode in pod {self.current_pod_name} (session {self.current_session_id})")

        try:
            while not done and step < self.max_steps_per_episode:
                # Get action from model (PIL Image input)
                action, generated_text = self._get_action(screenshot, self.task_prompt)
                # Execute action in environment
                next_screenshot, reward, done, info = self._execute_action(action)
                next_screenshot = self._preprocess_screenshot(next_screenshot)

                # Store transition (PIL Images directly)
                observations.append(screenshot)
                generated_texts.append(generated_text)
                rewards.append(reward)
                prompts.append(self.task_prompt)

                # Update state
                screenshot = next_screenshot
                step += 1

                logger.debug(f"Step {step}: action={action['action_type']}, reward={reward:.3f}, done={done}")

        finally:
            # Always delete pod, even if error occurred
            self._close_session(self.current_session_id, self.current_pod_name)

        # Create trajectory
        total_reward = sum(rewards)
        trajectory = Trajectory(
            observations=observations,
            generated_texts=generated_texts,
            rewards=rewards,
            prompts=prompts,
            metadata={
                'task_id': datetime.now().isoformat(),
                'session_id': self.current_session_id,
                'pod_name': self.current_pod_name,
                'success': done and total_reward > 0,
                'episode_length': step,
                'total_reward': total_reward,
                'complete': done or step >= self.max_steps_per_episode,
                'termination_reason': 'success' if done else 'max_steps'
            }
        )

        logger.info(f"Episode complete: {step} steps, reward={total_reward:.2f}")

        # Save raw data if configured
        if self.data_dir:
            self._save_trajectory(trajectory)

        return trajectory

    def _save_trajectory(self, trajectory: Trajectory):
        """Save trajectory to disk for future use."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = trajectory.metadata.get('task_id', 'unknown')
        session_id = trajectory.metadata.get('session_id', 'unknown')
        filename = f"traj_{timestamp}_session{session_id}.npz"
        filepath = self.data_dir / filename

        try:
            np.savez_compressed(filepath, **trajectory.to_dict())
            logger.debug(f"Saved trajectory to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}")
