import enum
import os
import asyncio
import pyautogui
from Xlib import display, X
from PIL import Image
from typing import Union, List


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


async def act(type: ActionType, **kwargs):
    """Execute an action based on the action type"""
    match type:
        case ActionType.Screenshot:
            return await screenshot()
        case ActionType.MouseMove:
            await mouse_move(x=kwargs["x"], y=kwargs["y"])
        case ActionType.LeftClick:
            await left_click(x=kwargs["x"], y=kwargs["y"])
        case ActionType.RightClick:
            await right_click(x=kwargs["x"], y=kwargs["y"])
        case ActionType.DoubleClick:
            await double_click(x=kwargs["x"], y=kwargs["y"])
        case ActionType.TripleClick:
            await triple_click(x=kwargs["x"], y=kwargs["y"])
        case ActionType.Type:
            await type_text(text=kwargs["text"])
        case ActionType.Keys:
            await keys(keys=kwargs["keys"])
        case ActionType.Scroll:
            await scroll(direction=kwargs["direction"], x=kwargs.get("x"), y=kwargs.get("y"))


async def screenshot():
    """Take a screenshot of the current screen"""
    disp = display.Display(os.environ["DISPLAY"])
    root = disp.screen().root
    geom = root.get_geometry()
    width = geom.width
    height = geom.height
    raw = root.get_image(0, 0, width, height, X.ZPixmap, 0xffffffff)
    return Image.frombytes("RGB", (width, height), raw.data, "raw", "BGRX")


async def mouse_move(x: int, y: int):
    """Move the mouse cursor to the specified coordinates"""
    try:
        await asyncio.to_thread(pyautogui.moveTo, x, y)
    except pyautogui.FailSafeException:
        raise RuntimeError("PyAutoGUI fail-safe triggered")
    except Exception as e:
        raise RuntimeError(f"Failed to move mouse to ({x}, {y}): {e}")


async def left_click(x: int, y: int):
    """Perform a left click at the specified coordinates"""
    try:
        await asyncio.to_thread(pyautogui.click, x, y, button='left')
    except pyautogui.FailSafeException:
        raise RuntimeError("PyAutoGUI fail-safe triggered")
    except Exception as e:
        raise RuntimeError(f"Failed to left click at ({x}, {y}): {e}")


async def right_click(x: int, y: int):
    """Perform a right click at the specified coordinates"""
    try:
        await asyncio.to_thread(pyautogui.click, x, y, button='right')
    except pyautogui.FailSafeException:
        raise RuntimeError("PyAutoGUI fail-safe triggered")
    except Exception as e:
        raise RuntimeError(f"Failed to right click at ({x}, {y}): {e}")


async def double_click(x: int, y: int):
    """Perform a double click at the specified coordinates"""
    try:
        await asyncio.to_thread(pyautogui.doubleClick, x, y)
    except pyautogui.FailSafeException:
        raise RuntimeError("PyAutoGUI fail-safe triggered")
    except Exception as e:
        raise RuntimeError(f"Failed to double click at ({x}, {y}): {e}")


async def triple_click(x: int, y: int):
    """Perform a triple click at the specified coordinates"""
    try:
        # PyAutoGUI doesn't have triple click, so we'll do it manually
        await asyncio.to_thread(lambda: pyautogui.click(x, y, clicks=3, interval=0.1))
    except pyautogui.FailSafeException:
        raise RuntimeError("PyAutoGUI fail-safe triggered")
    except Exception as e:
        raise RuntimeError(f"Failed to triple click at ({x}, {y}): {e}")


async def type_text(text: str):
    """Type the specified text"""
    try:
        await asyncio.to_thread(pyautogui.typewrite, text)
    except pyautogui.FailSafeException:
        raise RuntimeError("PyAutoGUI fail-safe triggered")
    except Exception as e:
        raise RuntimeError(f"Failed to type text '{text}': {e}")


async def keys(keys: Union[str, List[str]]):
    """Press the specified key(s)"""
    try:
        if isinstance(keys, str):
            # Split by '+' to handle combinations like "ctrl+shift+c"
            keys = [key.strip().lower() for key in keys.split('+')]
        
        await asyncio.to_thread(pyautogui.hotkey, *keys)
    except pyautogui.FailSafeException:
        raise RuntimeError("PyAutoGUI fail-safe triggered")
    except Exception as e:
        raise RuntimeError(f"Failed to press keys {keys}: {e}")


async def scroll(direction: str, x: int = None, y: int = None, amount: int = 3):
    """Scroll in the specified direction

    Args:
        direction: 'up' or 'down'
        x: Optional x coordinate to move mouse to before scrolling
        y: Optional y coordinate to move mouse to before scrolling
        amount: Number of scroll units (default: 3)
    """
    try:
        # Move mouse to position if coordinates are provided
        if x is not None and y is not None:
            await asyncio.to_thread(pyautogui.moveTo, x, y)

        # Scroll up (positive) or down (negative)
        scroll_amount = amount if direction.lower() == "up" else -amount
        await asyncio.to_thread(pyautogui.scroll, scroll_amount)
    except pyautogui.FailSafeException:
        raise RuntimeError("PyAutoGUI fail-safe triggered")
    except Exception as e:
        raise RuntimeError(f"Failed to scroll {direction}: {e}")