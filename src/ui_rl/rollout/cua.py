import enum
import logging
from dataclasses import dataclass
from PIL import Image
from .uitars import UITARSRollout
from .runtime import CUASessionRuntime


logger = logging.getLogger(__name__)


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
    Scroll = "scroll"

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


@dataclass
class Action:
    action_type: ActionType
    x: int | None = None
    y: int | None = None
    text: str | None = None
    keys: str | None = None
    direction: str | None = None


async def run_cua_rollout(
    rollout: UITARSRollout,
    runtime: CUASessionRuntime,
    max_steps: int = 10,
):
    """
    Launches a session pod, awaits it ready, and executes a Computer Use agent in it.
    """
    session_id = await runtime.create_session()
    try:
        logger.info(f"({session_id}) Starting...")
        await runtime.session_ready(session_id)
        logger.info(f"({session_id}) Ready")

        # Start with getting the init state (e.g. take screenshot)
        action = Action(ActionType.Screenshot)
        state = await runtime.session_act(session_id, action)

        for step_num in range(max_steps):
            logger.info(f"({session_id}) Predicting action {step_num+1}")
            action = await rollout.predict_next_action(state)
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
        runtime.teardown_session(session_id)
