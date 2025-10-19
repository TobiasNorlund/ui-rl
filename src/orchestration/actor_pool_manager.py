"""Actor Pool Manager for continuous trajectory collection with VM resource limits."""

import threading
import queue
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import torch.nn as nn

from ..actor.task_runner import TaskRunner

logger = logging.getLogger(__name__)


class ActorPoolManager:
    """
    Manages a pool of TaskRunner actors with VM resource constraints.

    Key features:
    - One actor runs one episode then exits (clean session lifecycle)
    - Maintains target number of concurrent actors
    - Respects max concurrent sessions per VM (memory constraint)
    - Automatically spawns replacement actors when episodes finish
    - Round-robin VM assignment for load balancing
    - Monitor thread for health checks and crash recovery

    Design:
    Each actor thread:
      1. Creates TaskRunner
      2. Runs exactly ONE episode (creates/closes VM session)
      3. Puts trajectory in queue
      4. Exits (thread terminates)
      5. Pool spawns replacement actor
    """

    def __init__(
        self,
        target_concurrent_actors: int,
        vm_urls: List[str],
        max_concurrent_per_vm: int,
        model: nn.Module,
        trajectory_queue: queue.Queue,
        task_prompt: str,
        session_type: str = "simple_data_entry",
        max_steps_per_episode: int = 20,
        action_format: str = "json",
        action_delay: float = 1.0,
        data_dir: Optional[Path] = None,
        monitor_interval: float = 2.0
    ):
        """
        Initialize Actor Pool Manager.

        Args:
            target_concurrent_actors: Target number of actors running concurrently
            vm_urls: List of VM URLs to distribute actors across
            max_concurrent_per_vm: Max concurrent sessions per VM (memory limit)
            model: VLM model for actors to use
            trajectory_queue: Queue to put collected trajectories
            task_prompt: Task prompt for actors
            session_type: Type of VM session to create
            max_steps_per_episode: Max steps per episode
            action_format: Action format for action decoder
            action_delay: Delay in seconds after each action
            data_dir: Optional directory to save raw trajectories
            monitor_interval: How often to check actor health (seconds)
        """
        self.target_concurrent_actors = target_concurrent_actors
        self.vm_urls = vm_urls
        self.max_concurrent_per_vm = max_concurrent_per_vm
        self.model = model
        self.trajectory_queue = trajectory_queue
        self.task_prompt = task_prompt
        self.session_type = session_type
        self.max_steps_per_episode = max_steps_per_episode
        self.action_format = action_format
        self.action_delay = action_delay
        self.data_dir = Path(data_dir) if data_dir else None
        self.monitor_interval = monitor_interval

        # State tracking (protected by lock)
        self.lock = threading.Lock()
        self.active_actors: Dict[int, Dict[str, Any]] = {}
        self.active_count_per_vm: Dict[str, int] = {url: 0 for url in vm_urls}

        # Actor ID counter
        self.next_actor_id = 1

        # Round-robin state for load balancing
        self.next_vm_index = 0

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
        logger.info(f"VMs: {len(vm_urls)}")
        logger.info(f"Max concurrent per VM: {max_concurrent_per_vm}")
        logger.info(f"Total capacity: {len(vm_urls) * max_concurrent_per_vm} concurrent sessions")
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

        # Spawn initial actors up to target (respecting VM limits)
        initial_spawned = 0
        for _ in range(self.target_concurrent_actors):
            vm_url = self._get_next_available_vm()
            if vm_url:
                self._spawn_actor(vm_url)
                initial_spawned += 1
            else:
                logger.warning(f"Could not spawn actor: all VMs at capacity")
                break

        logger.info(f"Spawned {initial_spawned} initial actors")

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

    def _get_next_available_vm(self) -> Optional[str]:
        """
        Get next VM that has capacity for another actor.
        Uses round-robin for load balancing to distribute load evenly.

        Returns:
            VM URL or None if all VMs are at capacity
        """
        with self.lock:
            # Try each VM starting from next_vm_index (round-robin)
            for i in range(len(self.vm_urls)):
                # Calculate actual index with wrap-around
                vm_index = (self.next_vm_index + i) % len(self.vm_urls)
                vm_url = self.vm_urls[vm_index]

                if self.active_count_per_vm[vm_url] < self.max_concurrent_per_vm:
                    # Found available VM, update index for next time
                    self.next_vm_index = (vm_index + 1) % len(self.vm_urls)
                    return vm_url

            # All VMs at capacity
            return None

    def _spawn_actor(self, vm_url: str) -> int:
        """
        Spawn a new actor thread on the specified VM.

        Args:
            vm_url: VM URL to spawn actor on

        Returns:
            Actor ID
        """
        with self.lock:
            actor_id = self.next_actor_id
            self.next_actor_id += 1

        # Create thread that runs ONE episode
        thread = threading.Thread(
            target=self._run_single_episode,
            args=(actor_id, vm_url),
            name=f"Actor-{actor_id}",
            daemon=False  # We want to wait for these to finish
        )

        # Register actor before starting
        with self.lock:
            self.active_actors[actor_id] = {
                'thread': thread,
                'vm_url': vm_url,
                'start_time': datetime.now(),
                'status': 'running'
            }
            self.active_count_per_vm[vm_url] += 1
            self.stats['total_actors_spawned'] += 1

        thread.start()
        logger.debug(f"Spawned Actor-{actor_id} on {vm_url} "
                    f"({self.active_count_per_vm[vm_url]}/{self.max_concurrent_per_vm})")

        return actor_id

    def _run_single_episode(self, actor_id: int, vm_url: str):
        """
        Run exactly ONE episode then exit.

        This is the target function for actor threads.

        Args:
            actor_id: Unique actor ID
            vm_url: VM URL to connect to
        """
        actor_logger = logging.getLogger(f"Actor-{actor_id}")
        actor_logger.info(f"Starting episode on {vm_url}")

        try:
            # Create TaskRunner for single episode
            runner = TaskRunner(
                ui_env_url=vm_url,
                model=self.model,
                task_prompt=self.task_prompt,
                session_type=self.session_type,
                max_steps_per_episode=self.max_steps_per_episode,
                action_delay=self.action_delay,
                action_format=self.action_format,
                data_dir=self.data_dir
            )

            # Run EXACTLY ONE episode
            # This creates session, runs episode, closes session
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
                    vm_url = info['vm_url']

                    # Free up VM slot
                    self.active_count_per_vm[vm_url] -= 1

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
                        vm_url = self._get_next_available_vm()
                        if vm_url:
                            self._spawn_actor(vm_url)
                        else:
                            # All VMs at capacity, wait for next cycle
                            break

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
            stats['active_per_vm'] = dict(self.active_count_per_vm)

            if stats['start_time']:
                elapsed = (datetime.now() - stats['start_time']).total_seconds()
                stats['uptime_seconds'] = elapsed
                if elapsed > 0:
                    stats['episodes_per_second'] = stats['total_episodes_collected'] / elapsed

            return stats
