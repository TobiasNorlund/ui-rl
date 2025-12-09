"""
Module for recording user input events (mouse clicks, scrolls, and keyboard presses).
"""
import time
import json
import threading
import base64
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
from pynput import mouse, keyboard
from pynput.mouse import Button, Listener as MouseListener
from pynput.keyboard import Key, Listener as KeyboardListener
from computer import ActionType, screenshot


class EventType(Enum):
    """Types of recorded events"""
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    MOUSE_SCROLL = "mouse_scroll"
    KEYBOARD_PRESS = "keyboard_press"
    KEYBOARD_RELEASE = "keyboard_release"
    SCREENSHOT = "screenshot"


@dataclass
class MouseClickEvent:
    """Represents a mouse click event"""
    event_type: str = EventType.MOUSE_CLICK.value
    button: str = ""  # 'left', 'right', 'middle'
    x: int = 0
    y: int = 0
    timestamp: float = 0.0
    pressed: bool = True  # True for press, False for release


@dataclass
class MouseMoveEvent:
    """Represents a mouse movement event"""
    event_type: str = EventType.MOUSE_MOVE.value
    x: int = 0
    y: int = 0
    timestamp: float = 0.0


@dataclass
class MouseScrollEvent:
    """Represents a mouse scroll event"""
    event_type: str = EventType.MOUSE_SCROLL.value
    x: int = 0
    y: int = 0
    dx: int = 0  # Horizontal scroll delta
    dy: int = 0  # Vertical scroll delta
    timestamp: float = 0.0


@dataclass
class KeyboardEvent:
    """Represents a keyboard event"""
    event_type: str = EventType.KEYBOARD_PRESS.value
    key: str = ""
    timestamp: float = 0.0
    pressed: bool = True  # True for press, False for release


@dataclass
class ProcessedAction:
    """Represents a processed high-level action"""
    action_type: str
    timestamp: float
    # For clicks
    x: Optional[int] = None
    y: Optional[int] = None
    # For typing
    text: Optional[str] = None
    # For hotkeys
    keys: Optional[List[str]] = None
    # For scrolling
    dx: Optional[int] = None
    dy: Optional[int] = None


class InputRecorder:
    """Records mouse clicks and keyboard presses"""
    
    def __init__(self, record_mouse_moves: bool = False):
        """
        Initialize the input recorder.
        
        Args:
            record_mouse_moves: If True, also record mouse movement events
        """
        self.record_mouse_moves = record_mouse_moves
        self.events: List[Dict] = []
        self.screenshots: List[Dict] = []
        self.start_time: Optional[float] = None
        self.mouse_listener: Optional[MouseListener] = None
        self.keyboard_listener: Optional[KeyboardListener] = None
        self.is_recording = False
        self._screenshot_thread: Optional[threading.Thread] = None
        self._stop_screenshot_thread = False
        self._last_check_time: Optional[float] = None
        
    def _get_timestamp(self) -> float:
        """Get current timestamp relative to recording start"""
        if self.start_time is None:
            return time.time()
        return time.time() - self.start_time
    
    def _take_screenshot(self) -> Optional[str]:
        """Take a screenshot and return it as base64 string"""
        try:
            img = screenshot()
            
            # Convert to base64
            buf = BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return None
    
    def _screenshot_check_loop(self):
        """Background thread that takes screenshots"""
        while not self._stop_screenshot_thread and self.is_recording:
            time.sleep(0.5)
            
            if not self.is_recording:
                break
            
            current_time = self._get_timestamp()
            check_start_time = current_time - 2.0
            
            # Check if there were any events in the last second
            events_in_last_second = (self.events or [{}])[-1].get('timestamp', 0) >= check_start_time

            if events_in_last_second:
                # Take a screenshot
                screenshot_b64 = self._take_screenshot()
                if screenshot_b64:
                    screenshot_entry = {
                        'timestamp': current_time,
                        'screenshot_base64': screenshot_b64
                    }
                    self.screenshots.append(screenshot_entry)            
            self._last_check_time = current_time
    
    def _on_mouse_click(self, x: int, y: int, button: Button, pressed: bool):
        """Callback for mouse click events"""
        if not self.is_recording:
            return
            
        button_name = "left" if button == Button.left else "right" if button == Button.right else "middle"
        
        event = MouseClickEvent(
            button=button_name,
            x=x,
            y=y,
            timestamp=self._get_timestamp(),
            pressed=pressed
        )
        self.events.append(asdict(event))
    
    def _on_mouse_move(self, x: int, y: int):
        """Callback for mouse movement events"""
        if not self.is_recording or not self.record_mouse_moves:
            return
            
        event = MouseMoveEvent(
            x=x,
            y=y,
            timestamp=self._get_timestamp()
        )
        self.events.append(asdict(event))
    
    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int):
        """Callback for mouse scroll events"""
        if not self.is_recording:
            return
            
        event = MouseScrollEvent(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            timestamp=self._get_timestamp()
        )
        self.events.append(asdict(event))
    
    def _on_key_press(self, key):
        """Callback for keyboard press events"""
        if not self.is_recording:
            return
            
        try:
            # Try to get the character representation
            key_str = key.char if hasattr(key, 'char') and key.char else str(key)
        except AttributeError:
            # For special keys, use the name
            key_str = str(key).replace('Key.', '')
        
        event = KeyboardEvent(
            key=key_str,
            timestamp=self._get_timestamp(),
            pressed=True
        )
        self.events.append(asdict(event))
    
    def _on_key_release(self, key):
        """Callback for keyboard release events"""
        if not self.is_recording:
            return
            
        try:
            key_str = key.char if hasattr(key, 'char') and key.char else str(key)
        except AttributeError:
            key_str = str(key).replace('Key.', '')
        
        event = KeyboardEvent(
            event_type=EventType.KEYBOARD_RELEASE.value,
            key=key_str,
            timestamp=self._get_timestamp(),
            pressed=False
        )
        self.events.append(asdict(event))
    
    def start(self):
        """Start recording input events"""
        if self.is_recording:
            return
        
        self.events.clear()
        self.screenshots.clear()
        self.start_time = time.time()
        self.is_recording = True
        self._stop_screenshot_thread = False
        self._last_check_time = None
        
        # Start mouse listener
        self.mouse_listener = MouseListener(
            on_click=self._on_mouse_click,
            on_move=self._on_mouse_move if self.record_mouse_moves else None,
            on_scroll=self._on_mouse_scroll
        )
        self.mouse_listener.start()
        
        # Start keyboard listener
        self.keyboard_listener = KeyboardListener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.keyboard_listener.start()
        
        # Start screenshot monitoring thread
        self._screenshot_thread = threading.Thread(target=self._screenshot_check_loop, daemon=True)
        self._screenshot_thread.start()
    
    def stop(self):
        """Stop recording input events"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self._stop_screenshot_thread = True
        
        # Wait for screenshot thread to finish (with timeout)
        if self._screenshot_thread and self._screenshot_thread.is_alive():
            self._screenshot_thread.join(timeout=2.0)
        self._screenshot_thread = None
        
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None


def process_events(events: List[Dict], 
                   double_click_timeout: float = 0.5,
                   typing_timeout: float = 0.5) -> List[Dict]:
    """
    Process raw events into high-level actions.
    
    Detects:
    - Single, double, and triple clicks
    - Consecutive typing
    - Hotkeys (key combinations)
    - Scrolling
    
    Args:
        events: List of raw event dictionaries
        double_click_timeout: Maximum time between clicks to count as double/triple click (seconds)
        typing_timeout: Maximum time between key presses to count as consecutive typing (seconds)
        
    Returns:
        List of processed action dictionaries
    """
    processed_actions: List[Dict] = []
    
    # Track state for processing
    i = 0
    click_buffer: deque = deque()  # Store recent clicks for double/triple detection
    typing_buffer: List[Dict] = []  # Store consecutive typing events
    typing_start_time: Optional[float] = None
    pressed_modifiers: Set[str] = set()  # Track currently pressed modifier keys
    pressed_keys: Set[str] = set()  # Track all currently pressed keys
    
    # Modifier key names
    MODIFIER_KEYS = {'ctrl', 'ctrl_l', 'ctrl_r', 'alt', 'alt_l', 'alt_r', 
                     'alt_gr', 'shift', 'shift_l', 'shift_r', 'cmd', 'cmd_l', 'cmd_r'}
    
    def is_modifier(key: str) -> bool:
        """Check if a key is a modifier"""
        key_lower = key.lower()
        return any(mod in key_lower for mod in MODIFIER_KEYS)
    
    def normalize_key(key: str) -> str:
        """Normalize key names for consistency"""
        key_lower = key.lower()
        if 'ctrl' in key_lower:
            return 'ctrl'
        elif 'alt' in key_lower:
            return 'alt'
        elif 'shift' in key_lower:
            return 'shift'
        elif 'cmd' in key_lower or 'super' in key_lower:
            return 'cmd'
        return key_lower
    
    def flush_typing():
        """Flush accumulated typing into a single action"""
        nonlocal typing_buffer, typing_start_time
        if typing_buffer:
            text = ''.join(evt.get('key', '') for evt in typing_buffer 
                          if len(evt.get('key', '')) == 1)
            if text:
                action = ProcessedAction(
                    action_type=ActionType.Type.value,
                    timestamp=typing_start_time or typing_buffer[0].get('timestamp', 0),
                    text=text
                )
                processed_actions.append(asdict(action))
            typing_buffer = []
            typing_start_time = None
    
    while i < len(events):
        event = events[i]
        event_type = event.get('event_type', '')
        timestamp = event.get('timestamp', 0)
        key = event.get('key', '')
        pressed = event.get('pressed', True)
        
        # Handle keyboard events
        if event_type in (EventType.KEYBOARD_PRESS.value, EventType.KEYBOARD_RELEASE.value):
            key_normalized = normalize_key(key)
            is_mod = is_modifier(key)
            
            if pressed:
                pressed_keys.add(key)
                if is_mod:
                    pressed_modifiers.add(key_normalized)
                
                # Check if this is a hotkey (modifier + non-modifier key)
                if pressed_modifiers and not is_mod:
                    # This is a hotkey combination (modifier + any non-modifier key)
                    # Collect all modifiers and the key
                    key_name = key.lower() if len(key) == 1 else key.lower().replace('key.', '')
                    hotkey_parts = sorted(pressed_modifiers) + [key_name]
                    action = ProcessedAction(
                        action_type=ActionType.Keys.value,
                        timestamp=timestamp,
                        keys=hotkey_parts
                    )
                    processed_actions.append(asdict(action))
                    # Clear typing buffer since this interrupts typing
                    flush_typing()
                elif not is_mod and len(key) == 1:
                    # Regular character (no modifiers) - check if we should flush previous typing
                    if typing_buffer and typing_start_time:
                        time_since_last = timestamp - typing_buffer[-1].get('timestamp', typing_start_time)
                        if time_since_last > typing_timeout:
                            # Gap too long, flush previous typing
                            flush_typing()
                    
                    # Add to typing buffer
                    if not typing_buffer:
                        typing_start_time = timestamp
                    typing_buffer.append(event)
                elif not is_mod:
                    # Special key (not modifier, not character) without modifiers
                    # Just flush typing, don't create action (could be arrow keys, etc.)
                    flush_typing()
            else:
                # Key release
                pressed_keys.discard(key)
                if is_mod:
                    pressed_modifiers.discard(key_normalized)
                    # If all modifiers released, flush typing
                    if not pressed_modifiers:
                        flush_typing()
        
        # Handle mouse click events
        elif event_type == EventType.MOUSE_CLICK.value:
            button = event.get('button', '')
            x = event.get('x', 0)
            y = event.get('y', 0)
            
            if pressed:  # Only process press events, ignore releases
                # Flush any pending typing
                flush_typing()
                
                # Add to click buffer
                click_buffer.append({
                    'button': button,
                    'x': x,
                    'y': y,
                    'timestamp': timestamp
                })
                
                # Keep only recent clicks of the same button within timeout window
                click_buffer = deque([c for c in click_buffer 
                                     if c['button'] == button and 
                                     (timestamp - c['timestamp']) <= double_click_timeout])
                
                # Check if there's another click of the same button coming soon
                lookahead_time = double_click_timeout
                found_next_click = False
                next_event_time = None
                
                # Look ahead to find next click of same button
                for j in range(i + 1, len(events)):
                    next_event = events[j]
                    next_ts = next_event.get('timestamp', 0)
                    
                    # Stop looking if we've gone too far in time
                    if (next_ts - timestamp) > lookahead_time:
                        break
                    
                    # Check if this is a click of the same button
                    if (next_event.get('event_type') == EventType.MOUSE_CLICK.value and
                        next_event.get('button') == button and
                        next_event.get('pressed', True)):
                        found_next_click = True
                        next_event_time = next_ts
                        break
                
                # Process clicks if no more clicks coming soon, or if we have 3+ clicks
                should_process = False
                if len(click_buffer) >= 3:
                    # Definitely a triple click (or more)
                    should_process = True
                elif not found_next_click:
                    # No more clicks coming, process what we have
                    should_process = True
                elif found_next_click and next_event_time:
                    # There's another click coming, but if we already have 2 clicks
                    # and the next one is far enough, we might have a double
                    # Actually, wait for the next click
                    should_process = False
                
                if should_process:
                    if len(click_buffer) >= 3:
                        # Triple click (or more - take the last 3)
                        clicks_to_process = list(click_buffer)[-3:]
                        last_click = clicks_to_process[-1]
                        action = ProcessedAction(
                            action_type=ActionType.TripleClick.value,
                            timestamp=last_click['timestamp'],
                            x=last_click['x'],
                            y=last_click['y'],
                        )
                        processed_actions.append(asdict(action))
                        click_buffer.clear()
                    elif len(click_buffer) >= 2:
                        # Double click
                        last_click = click_buffer[-1]
                        action = ProcessedAction(
                            action_type=ActionType.DoubleClick.value,
                            timestamp=last_click['timestamp'],
                            x=last_click['x'],
                            y=last_click['y'],
                        )
                        processed_actions.append(asdict(action))
                        click_buffer.clear()
                    else:
                        # Single click
                        click = click_buffer[-1]
                        if button == 'left':
                            action = ProcessedAction(
                                action_type=ActionType.LeftClick.value,
                                timestamp=click['timestamp'],
                                x=click['x'],
                                y=click['y'],
                            )
                        elif button == 'right':
                            action = ProcessedAction(
                                action_type=ActionType.RightClick.value,
                                timestamp=click['timestamp'],
                                x=click['x'],
                                y=click['y'],
                            )
                        else:
                            # Unsupported button
                            continue
                        processed_actions.append(asdict(action))
                        click_buffer.clear()
        
        # Handle scroll events
        elif event_type == EventType.MOUSE_SCROLL.value:
            # Check if typing should be flushed due to time gap
            if typing_buffer and typing_start_time:
                time_since_last = timestamp - typing_buffer[-1].get('timestamp', typing_start_time)
                if time_since_last > typing_timeout:
                    flush_typing()
            else:
                flush_typing()
            
            action = ProcessedAction(
                action_type=ActionType.Scroll.value,
                timestamp=timestamp,
                x=event.get('x', 0),
                y=event.get('y', 0),
                dx=event.get('dx', 0),
                dy=event.get('dy', 0)
            )
            processed_actions.append(asdict(action))
        
        # Handle other events (mouse move) - check typing timeout
        else:
            if typing_buffer and typing_start_time:
                time_since_last = timestamp - typing_buffer[-1].get('timestamp', typing_start_time)
                if time_since_last > typing_timeout:
                    flush_typing()
        
        i += 1
    
    # Flush any remaining typing
    flush_typing()
    
    # Process any remaining clicks in buffer
    if click_buffer:
        if len(click_buffer) >= 3:
            last_click = click_buffer[-1]
            action = ProcessedAction(
                action_type=ActionType.TripleClick.value,
                timestamp=last_click['timestamp'],
                x=last_click['x'],
                y=last_click['y'],
                button=last_click['button']
            )
            processed_actions.append(asdict(action))
        elif len(click_buffer) >= 2:
            last_click = click_buffer[-1]
            action = ProcessedAction(
                action_type=ActionType.DoubleClick.value,
                timestamp=last_click['timestamp'],
                x=last_click['x'],
                y=last_click['y'],
                button=last_click['button']
            )
            processed_actions.append(asdict(action))
        else:
            click = click_buffer[0]
            if click['button'] == 'left':
                action = ProcessedAction(
                    action_type=ActionType.LeftClick.value,
                    timestamp=click['timestamp'],
                    x=click['x'],
                    y=click['y'],
                    button=click['button']
                )
            elif click['button'] == 'right':
                action = ProcessedAction(
                    action_type=ActionType.RightClick.value,
                    timestamp=click['timestamp'],
                    x=click['x'],
                    y=click['y'],
                    button=click['button']
                )
            else:
                action = ProcessedAction(
                    action_type=ActionType.LeftClick.value,
                    timestamp=click['timestamp'],
                    x=click['x'],
                    y=click['y'],
                    button=click['button']
                )
            processed_actions.append(asdict(action))
    
    return processed_actions


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(f"./{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))
    args = parser.parse_args()

    print("Recording input events. Press Ctrl+C to stop...")
    recorder = InputRecorder(record_mouse_moves=False)
    recorder.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
    print(f"\nRecorded {len(recorder.events)} raw events")
    
    # Process events
    processed = process_events(recorder.events)
    print(f"Processed into {len(processed)} computer use events")
    
    with args.output.open("w") as f:
        json.dump({
            "events": processed,
            "screenshots": recorder.screenshots
        }, f)
    print(f"Saved recording to {str(args.output)}")
