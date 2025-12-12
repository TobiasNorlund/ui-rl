import logging

from .task import TaskSpec
from .runtime import CUASessionRuntime
from .cua import Action, ActionType
from .models.uitars15 import UITARS15_Rollout


logger = logging.getLogger(__name__)


async def run_cua_rollout(
    task_spec: TaskSpec,
    rollout: UITARS15_Rollout,
    runtime: CUASessionRuntime,
    max_steps: int = 10,
):
    """
    Launches a session, awaits it ready, and executes a Computer Use agent in it.
    """
    session_id = await task_spec.create_session(runtime)
    try:
        logger.info(f"({session_id}) Starting...")
        await runtime.session_ready(session_id)
        logger.info(f"({session_id}) Ready")

        # Start with getting the init state (e.g. take screenshot)
        action  = Action(ActionType.Screenshot)
        state = await runtime.session_act(session_id, action)

        for step_num in range(max_steps):
            logger.info(f"({session_id}) Predicting action {step_num+1}")
            action = await rollout.predict_next_action(state)  # type: ignore
            if action is None:
                break
                
            logger.info(f"({session_id}) Taking action: {action}")
            state = await runtime.session_act(session_id, action)

        # Get final rollout progress
        progress = await runtime.get_session_progress(session_id)
        rollout.progress = progress

        logger.info(f"({session_id}) Finished successfully")
        return rollout
    except Exception as e:
        logger.error(f"({session_id}): {str(e)}")
        raise
    finally:
        await runtime.teardown_session(session_id)
