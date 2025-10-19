"""TaskRunner for executing UI tasks and collecting trajectories using ui-verifiers API."""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import requests
from PIL import Image
from io import BytesIO

from ..data_utils.trajectory import Trajectory
from ..data_utils.actions_decoder import ActionsDecoder

logger = logging.getLogger(__name__)


class TaskRunner:
    """
    Actor component: Runs episodes in the UI environment and collects trajectories.

    This implementation uses the ui-verifiers API for UI interaction:
    - POST /session: Create a new session
    - GET /session/{id}/screenshot: Get current screenshot
    - GET /session/{id}/act: Perform action and get resulting screenshot
    - GET /session/{id}/progress: Get reward/progress feedback
    - DELETE /session/{id}: Close session

    Design decisions:
    1. Maintains connection to remote UI environment (ui-verifiers FastAPI)
    2. Has reference to model for inference (shared with Trainer initially)
    3. Collects full trajectories before sending to training
    4. Handles screenshot preprocessing
    5. Uses ActionsDecoder to parse VLM outputs into API actions

    Responsibilities:
    - Create UI environment sessions
    - Run inference with VLM to get actions
    - Execute actions via ui-verifiers API
    - Collect rewards from progress endpoint
    - Build trajectory and send to queue when complete
    """

    def __init__(
        self,
        ui_env_url: str,
        model: nn.Module,
        task_prompt: str,
        session_type: str = "simple_data_entry",
        max_steps_per_episode: int = 50,
        screenshot_size: tuple = (224, 224),
        data_dir: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        action_format: str = "json",
        action_delay: float = 1.0,
    ):
        """
        Args:
            ui_env_url: URL of ui-verifiers FastAPI service
            model: VLM for action inference (shared reference with Trainer)
            task_prompt: Task description to send to VLM
            session_type: Type of UI session (e.g., "simple_data_entry")
            max_steps_per_episode: Maximum steps before episode terminates
            screenshot_size: Resize screenshots to this size (H, W)
            data_dir: Optional directory to save raw trajectories
            device: Device for model inference
            action_format: Format for action decoding ("json", "text", "coordinates", "natural")
            action_delay: Delay in seconds after each action (for UI to update)
        """
        self.ui_env_url = ui_env_url.rstrip('/')
        self.model = model
        self.model.eval()  # Important: set to eval mode for inference
        self.task_prompt = task_prompt
        self.session_type = session_type
        self.max_steps_per_episode = max_steps_per_episode
        self.screenshot_size = screenshot_size
        self.data_dir = Path(data_dir) if data_dir else None
        self.device = device
        self.action_delay = action_delay

        # Action decoder for parsing VLM outputs
        self.actions_decoder = ActionsDecoder(default_format=action_format)

        # State for current episode
        self.current_session_id: Optional[int] = None
        self.step_count = 0

        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TaskRunner initialized with env: {ui_env_url}")

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
        Send action to UI environment via ui-verifiers API and get next state.

        Args:
            action: Action dict to execute

        Returns:
            (next_screenshot, reward, done, info)
        """
        try:
            # Convert action to API parameters
            params = self.actions_decoder.action_to_api_params(action)
            params['delay'] = self.action_delay

            # Execute action via API
            response = requests.get(
                f"{self.ui_env_url}/session/{self.current_session_id}/act",
                params=params,
                timeout=30
            )
            response.raise_for_status()

            # Response is PNG image of screenshot after action
            screenshot = Image.open(BytesIO(response.content))

            # Get progress/reward from separate endpoint
            #progress_response = requests.get(
            #    f"{self.ui_env_url}/session/{self.current_session_id}/progress",
            #    timeout=30
            #)
            #logger.info(f"Progress: {progress_response}")
            #progress_response.raise_for_status()
            #progress_data = progress_response.json()
            

            # Calculate reward based on progress
            # This is task-specific; for simple_data_entry it tracks correct submissions
            #reward = self._calculate_reward(progress_data) # TODO: Protocol for progress not decided yet
            reward = 1.0 # TODO: Protocol for progress not decided yet

            # Check if episode should end
            #done = self._check_done(progress_data) # TODO: Protocol for done is not decided yet
            done = True # TODO: Protocol for done is not decided yet

            #return screenshot, reward, done, progress_data
            return screenshot, reward, done, {}

        except requests.exceptions.RequestException as e:
            logger.error(f"Error executing action: {e}")
            # Return dummy values and mark as done due to error
            dummy_screenshot = Image.new('RGB', self.screenshot_size, color='black')
            return dummy_screenshot, -1.0, True, {"error": str(e)}

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

    def _create_session(self) -> int:
        """
        Create a new UI session via ui-verifiers API.

        Returns:
            Session ID
        """
        try:
            response = requests.post(
                f"{self.ui_env_url}/session",
                params={
                    "type": self.session_type,
                    "n": 1
                },
                timeout=60  # Session creation can take longer
            )
            response.raise_for_status()

            data = response.json()
            session_id = data['session_ids'][0]

            logger.info(f"Created session {session_id}")
            return session_id

        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating session: {e}")
            raise

    def _close_session(self, session_id: int):
        """
        Close a UI session via ui-verifiers API.

        Args:
            session_id: Session ID to close
        """
        try:
            response = requests.delete(
                f"{self.ui_env_url}/session/{session_id}",
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"Closed session {session_id}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error closing session {session_id}: {e}")

    def _get_screenshot(self, session_id: int) -> Image.Image:
        """
        Get screenshot from session via ui-verifiers API.

        Args:
            session_id: Session ID

        Returns:
            PIL Image of screenshot
        """
        try:
            response = requests.get(
                f"{self.ui_env_url}/session/{session_id}/screenshot",
                timeout=30
            )
            response.raise_for_status()

            screenshot = Image.open(BytesIO(response.content))
            
            #screenshot.save(f"debug_screenshot_{session_id}.png")
            #logger.info(f"Screenshot saved to debug_screenshot_{session_id}.png")
            
            return screenshot

        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting screenshot: {e}")
            # Return blank image on error
            return Image.new('RGB', self.screenshot_size, color='black')

    def run_episode(self) -> Trajectory:
        """
        Run a single episode and return the complete trajectory.

        Returns:
            Completed trajectory with PIL Images in observations
        """
        # Create new session
        self.current_session_id = self._create_session()

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

        logger.info(f"Starting episode in session {self.current_session_id}")

        try:
            while not done and step < self.max_steps_per_episode:
                # Get action from model (PIL Image input)
                action, generated_text = (screenshot, self.task_prompt)
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
            # Always close session, even if error occurred
            self._close_session(self.current_session_id)
            #TODO: Handle logic if session ends with error

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
                'session_type': self.session_type,
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
