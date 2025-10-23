"""ActionsDecoder for converting VLM text output to UI actions.

This module provides a decoder that converts structured text output from
Vision-Language Models into action dictionaries compatible with the ui-verifiers API.

Supported action formats:
1. JSON format: {"action": "left_click", "x": 100, "y": 200}
2. Text format: left_click(100, 200)
3. Coordinate-first format: 100 200 left_click
4. Natural language format (experimental): "click at coordinates 100, 200"
"""

from typing import Dict, Any, Optional
import json
import re
import logging

logger = logging.getLogger(__name__)


class ActionsDecoder:
    """
    Decoder for converting VLM text output into UI actions.

    The decoder supports multiple input formats and converts them into
    action dictionaries compatible with the ui-verifiers API.

    Action types (from ui-verifiers):
    - screenshot: Take a screenshot (no parameters)
    - mouse_move: Move mouse to coordinates (x, y)
    - left_click: Left click at coordinates (x, y)
    - right_click: Right click at coordinates (x, y)
    - double_click: Double click at coordinates (x, y)
    - triple_click: Triple click at coordinates (x, y)
    """

    # Valid action types from ui-verifiers
    VALID_ACTIONS = {
        'screenshot',
        'mouse_move',
        'left_click',
        'right_click',
        'double_click',
        'triple_click',
    }


    def __init__(self, default_format: str = "json"):
        """
        Initialize the ActionsDecoder.

        Args:
            default_format: Default parsing format to try first.
                          Options: "json", "text", "coordinates", "natural"
        """
        self.default_format = default_format

    def decode(self, text: str) -> Dict[str, Any]:
        """
        Decode VLM text output into an action dictionary.

        Args:
            text: Generated text from VLM

        Returns:
            Action dictionary with keys:
            - action_type: The action type (e.g., "left_click")
            - x: X coordinate (if applicable)
            - y: Y coordinate (if applicable)
            - error: Error message (if parsing failed)

        Examples:
            >>> decoder = ActionsDecoder()
            >>> decoder.decode('{"action": "left_click", "x": 100, "y": 200}')
            {'action_type': 'left_click', 'x': 100, 'y': 200}

            >>> decoder.decode('left_click(100, 200)')
            {'action_type': 'left_click', 'x': 100, 'y': 200}

            >>> decoder.decode('100 200 left_click')
            {'action_type': 'left_click', 'x': 100, 'y': 200}
        """
        text = text.strip()

        # Try default format first
        if self.default_format == "json":
            result = self._decode_json(text)
            if "error" not in result:
                return result
        elif self.default_format == "text":
            result = self._decode_text(text)
            if "error" not in result:
                return result
        elif self.default_format == "coordinates":
            result = self._decode_coordinates(text)
            if "error" not in result:
                return result
        elif self.default_format == "natural":
            result = self._decode_natural_language(text)
            if "error" not in result:
                return result

        # Try all other formats
        for decode_func in [self._decode_json, self._decode_text,
                           self._decode_coordinates, self._decode_natural_language]:
            result = decode_func(text)
            if "error" not in result:
                return result

        # All parsing attempts failed
        logger.warning(f"Failed to parse action from text: {text}")
        return {
            "action_type": "screenshot",  # Fallback to safe action
            "error": f"Failed to parse action: {text}"
        }

    def _decode_json(self, text: str) -> Dict[str, Any]:
        """
        Decode JSON format: {"action": "left_click", "x": 100, "y": 200}
        """
        try:
            # Try to find JSON in text
            json_match = re.search(r'\{[^{}]*\}', text)
            if not json_match:
                return {"error": "No JSON found"}

            data = json.loads(json_match.group(0))

            # Extract action type
            action = data.get("action") or data.get("action_type") or data.get("type")
            if not action:
                return {"error": "No action field in JSON"}

            # Normalize action name
            action = self._normalize_action(action)
            if not action:
                return {"error": f"Invalid action type: {data.get('action')}"}

            result = {"action_type": action}

            # Add coordinates if present
            if "x" in data and "y" in data:
                result["x"] = int(data["x"])
                result["y"] = int(data["y"])

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return {"error": f"JSON decode error: {e}"}

    def _decode_text(self, text: str) -> Dict[str, Any]:
        """
        Decode text format: left_click(100, 200) or mouse_move(50, 75)
        """
        # Match pattern: action_name(x, y) or action_name()
        match = re.match(r'(\w+)\s*\(([^)]*)\)', text.strip())

        if not match:
            return {"error": "No text format match"}

        action_name = match.group(1)
        params_str = match.group(2).strip()

        # Normalize action name
        action = self._normalize_action(action_name)
        if not action:
            return {"error": f"Invalid action: {action_name}"}

        result = {"action_type": action}

        # Parse parameters if present
        if params_str:
            # Split by comma and extract numbers
            params = [p.strip() for p in params_str.split(',')]
            try:
                if len(params) >= 2:
                    result["x"] = int(params[0])
                    result["y"] = int(params[1])
            except (ValueError, IndexError):
                return {"error": f"Invalid parameters: {params_str}"}

        return result

    def _decode_coordinates(self, text: str) -> Dict[str, Any]:
        """
        Decode coordinate-first format: 100 200 left_click or 50 75 click
        """
        # Match pattern: number number action_name
        match = re.search(r'(\d+)\s+(\d+)\s+(\w+)', text)

        if not match:
            return {"error": "No coordinate format match"}

        x = int(match.group(1))
        y = int(match.group(2))
        action_name = match.group(3)

        # Normalize action name
        action = self._normalize_action(action_name)
        if not action:
            return {"error": f"Invalid action: {action_name}"}

        return {
            "action_type": action,
            "x": x,
            "y": y
        }

    def _decode_natural_language(self, text: str) -> Dict[str, Any]:
        """
        Decode natural language format (experimental):
        - "click at 100, 200"
        - "move mouse to coordinates 50, 75"
        - "take a screenshot"
        - "right click at position (100, 200)"
        """
        text_lower = text.lower()

        # Check for screenshot
        if any(word in text_lower for word in ['screenshot', 'capture', 'screen']):
            return {"action_type": "screenshot"}

        # Extract coordinates
        coord_patterns = [
            r'(\d+)\s*,\s*(\d+)',           # "100, 200"
            r'\((\d+)\s*,\s*(\d+)\)',       # "(100, 200)"
            r'x[:\s]*(\d+)\s*y[:\s]*(\d+)', # "x: 100 y: 200"
        ]

        coords = None
        for pattern in coord_patterns:
            match = re.search(pattern, text)
            if match:
                coords = (int(match.group(1)), int(match.group(2)))
                break

        if not coords:
            return {"error": "No coordinates found in natural language"}

        # Determine action type from keywords
        action = None
        if any(word in text_lower for word in ['double', 'double-click', 'doubleclick']):
            action = "double_click"
        elif any(word in text_lower for word in ['triple', 'triple-click', 'tripleclick']):
            action = "triple_click"
        elif any(word in text_lower for word in ['right', 'right-click', 'rightclick']):
            action = "right_click"
        elif any(word in text_lower for word in ['click', 'left-click', 'leftclick']):
            action = "left_click"
        elif any(word in text_lower for word in ['move', 'hover', 'goto', 'go to']):
            action = "mouse_move"
        else:
            # Default to left click if coordinates are present
            action = "left_click"

        return {
            "action_type": action,
            "x": coords[0],
            "y": coords[1]
        }

    def _normalize_action(self, action: str) -> Optional[str]:
        """
        Normalize action name to valid ui-verifiers action type.

        Args:
            action: Action name from parsed text

        Returns:
            Normalized action name, or None if invalid
        """
        action = action.lower().strip().replace('-', '_')

        # Check if already valid
        if action in self.VALID_ACTIONS:
            return action

        return None

    def validate_action(self, action: Dict[str, Any]) -> bool:
        """
        Validate that an action dictionary is properly formed.

        Args:
            action: Action dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        if "action_type" not in action:
            return False

        if action["action_type"] not in self.VALID_ACTIONS:
            return False

        # Actions other than screenshot require coordinates
        if action["action_type"] != "screenshot":
            if "x" not in action or "y" not in action:
                return False

            # Validate coordinates are integers
            if not isinstance(action["x"], int) or not isinstance(action["y"], int):
                return False

        return True

    def action_to_api_params(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert action dictionary to ui-verifiers API parameters.

        Args:
            action: Action dictionary from decode()

        Returns:
            Dictionary with query parameters for API call

        Example:
            >>> decoder.action_to_api_params({'action_type': 'left_click', 'x': 100, 'y': 200})
            {'action_type': 'left_click', 'x': 100, 'y': 200}
        """
        params = {"action_type": action["action_type"]}

        if "x" in action:
            params["x"] = action["x"]
        if "y" in action:
            params["y"] = action["y"]

        return params
