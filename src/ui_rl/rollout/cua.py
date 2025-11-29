import enum
from dataclasses import dataclass
from PIL import Image


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
