"""Actor Pool Manager for continuous trajectory collection with Kubernetes pods."""

import threading
import queue
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import torch.nn as nn

from ..actor.task_runner import TaskRunner

logger = logging.getLogger(__name__)


class ActorPoolManager:
    """
    Manages a pool of TaskRunner actors that create Kubernetes pods.

    Key features:
    - One actor runs one episode then exits (clean session lifecycle)
    - Maintains target number of concurrent actors
    - Automatically spawns replacement actors when episodes finish
    - Monitor thread for health checks and crash recovery
    - Kubernetes handles pod scheduling and resource management

    Design:
    Each actor thread:
      1. Creates TaskRunner
      2. Runs exactly ONE episode (TaskRunner creates/deletes Kubernetes pod)
      3. Puts trajectory in queue
      4. Exits (thread terminates)
      5. Pool spawns replacement actor
    """

    def __init__(
        self,
        target_concurrent_actors: int,
        cluster_host: str,
        pod_manifest_fn: Callable[[str, str], Dict],
        model: nn.Module,
        trajectory_queue: queue.Queue,
        task_prompt: str,
        namespace: str = "default",
        max_steps_per_episode: int = 20,
        action_format: str = "json",
        data_dir: Optional[Path] = None,
        monitor_interval: float = 2.0,
        session_timeout: int = 300,
    ):
        """
        Initialize Actor Pool Manager.

        Args:
            target_concurrent_actors: Target number of actors running concurrently
            cluster_host: Kubernetes cluster proxy server host/IP
            pod_manifest_fn: Function that generates pod manifest given (pod_name, session_id)
            model: VLM model for actors to use
            trajectory_queue: Queue to put collected trajectories
            task_prompt: Task prompt for actors
            namespace: Kubernetes namespace for pods
            max_steps_per_episode: Max steps per episode
            action_format: Action format for action decoder
            data_dir: Optional directory to save raw trajectories
            monitor_interval: How often to check actor health (seconds)
            session_timeout: Timeout in seconds for pod to come online
        """
        self.target_concurrent_actors = target_concurrent_actors
        self.cluster_host = cluster_host
        self.pod_manifest_fn = pod_manifest_fn
        self.namespace = namespace
        self.session_timeout = session_timeout
        self.model = model
        self.trajectory_queue = trajectory_queue
        self.task_prompt = task_prompt
        self.max_steps_per_episode = max_steps_per_episode
        self.action_format = action_format
        self.data_dir = Path(data_dir) if data_dir else None
        self.monitor_interval = monitor_interval

        # State tracking (protected by lock)
        self.lock = threading.Lock()
        self.active_actors: Dict[int, Dict[str, Any]] = {}

        # Actor ID counter
        self.next_actor_id = 1

        # Control
        self.stop_event = threading.Event()
        self.monitor_thread: Optional[threading.Thread] = None

        # Statistics
        self.stats = {
            'total_episodes_collected': 0,
            'total_actors_spawned': 0,
            'actors_crashed': 0,
            'start_time': None,
        }

        logger.info("=" * 60)
        logger.info("ActorPoolManager initialized")
        logger.info(f"Target concurrent actors: {target_concurrent_actors}")
        logger.info(f"Cluster host: {cluster_host}")
        logger.info(f"Namespace: {namespace}")
        logger.info("=" * 60)

    def start(self):
        """Start the actor pool."""
        logger.info("Starting ActorPoolManager...")

        with self.lock:
            self.stats['start_time'] = datetime.now()

        # Start monitor thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ActorPoolMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Monitor thread started")

        # Spawn initial actors up to target
        for _ in range(self.target_concurrent_actors):
            self._spawn_actor()

        logger.info(f"Spawned {self.target_concurrent_actors} initial actors")

    def stop(self, timeout: float = 30.0):
        """
        Stop the actor pool gracefully.

        Args:
            timeout: Max time to wait for actors to finish (seconds)
        """
        logger.info("Stopping ActorPoolManager...")

        # Signal stop (no new actors will be spawned)
        self.stop_event.set()

        # Wait for active actors to finish their current episode
        start_time = time.time()
        while True:
            with self.lock:
                active_count = len(self.active_actors)

            if active_count == 0:
                logger.info("All actors finished")
                break

            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(f"Timeout: {active_count} actors still running after {timeout}s")
                break

            logger.debug(f"Waiting for {active_count} actors to finish...")
            time.sleep(1)

        # Wait for monitor thread
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("ActorPoolManager stopped")

    def _spawn_actor(self) -> int:
        """
        Spawn a new actor thread.

        Returns:
            Actor ID
        """
        with self.lock:
            actor_id = self.next_actor_id
            self.next_actor_id += 1

        # Create thread that runs ONE episode
        thread = threading.Thread(
            target=self._run_single_episode,
            args=(actor_id,),
            name=f"Actor-{actor_id}",
            daemon=False  # We want to wait for these to finish
        )

        # Register actor before starting
        with self.lock:
            self.active_actors[actor_id] = {
                'thread': thread,
                'start_time': datetime.now(),
                'status': 'running'
            }
            self.stats['total_actors_spawned'] += 1

        thread.start()
        logger.debug(f"Spawned Actor-{actor_id}")

        return actor_id

    def _run_single_episode(self, actor_id: int):
        """
        Run exactly ONE episode then exit.

        This is the target function for actor threads.

        Args:
            actor_id: Unique actor ID
        """
        actor_logger = logging.getLogger(f"Actor-{actor_id}")
        actor_logger.info(f"Starting episode")

        try:
            # Create TaskRunner for single episode
            runner = TaskRunner(
                cluster_host=self.cluster_host,
                pod_manifest_fn=self.pod_manifest_fn,
                model=self.model,
                task_prompt=self.task_prompt,
                namespace=self.namespace,
                max_steps_per_episode=self.max_steps_per_episode,
                action_format=self.action_format,
                data_dir=self.data_dir,
                session_timeout=self.session_timeout
            )

            # Run EXACTLY ONE episode
            # This creates pod, runs episode, deletes pod
            trajectory = runner.run_episode()
            self.trajectory_queue.put(trajectory)

            actor_logger.info(f"Episode completed: {len(trajectory.observations)} steps, "
                            f"reward={trajectory.total_reward():.2f}")

            # Mark as finished successfully
            with self.lock:
                if actor_id in self.active_actors:
                    self.active_actors[actor_id]['status'] = 'finished'

        except Exception as e:
            actor_logger.error(f"Episode crashed: {e}", exc_info=True)

            # Mark as crashed
            with self.lock:
                if actor_id in self.active_actors:
                    self.active_actors[actor_id]['status'] = 'crashed'
                self.stats['actors_crashed'] += 1

        # Thread exits here (actor is done)

    def _monitor_loop(self):
        """
        Monitor thread that:
        - Checks actor health
        - Cleans up finished actors
        - Spawns replacement actors
        """
        logger.info("Monitor loop started")

        while not self.stop_event.is_set():
            time.sleep(self.monitor_interval)

            with self.lock:
                # Find finished/crashed actors (threads no longer alive)
                finished_actors = [
                    actor_id for actor_id, info in self.active_actors.items()
                    if not info['thread'].is_alive()
                ]

                # Clean up finished actors
                for actor_id in finished_actors:
                    info = self.active_actors.pop(actor_id)

                    # Update stats
                    if info['status'] == 'finished':
                        self.stats['total_episodes_collected'] += 1
                        logger.debug(f"Actor-{actor_id} finished successfully")
                    else:
                        logger.warning(f"Actor-{actor_id} crashed")

                # Spawn replacement actors if not stopping
                if not self.stop_event.is_set():
                    current_active = len(self.active_actors)
                    needed = self.target_concurrent_actors - current_active

                    for _ in range(needed):
                        self._spawn_actor()

        logger.info("Monitor loop stopped")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the actor pool.

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            stats = dict(self.stats)
            stats['active_actors'] = len(self.active_actors)

            if stats['start_time']:
                elapsed = (datetime.now() - stats['start_time']).total_seconds()
                stats['uptime_seconds'] = elapsed
                if elapsed > 0:
                    stats['episodes_per_second'] = stats['total_episodes_collected'] / elapsed

            return stats
