from typing import Literal, Any

from PIL import Image
from mimetypes import types_map
from pydantic import BaseModel, Field
from io import BytesIO
import base64


SYSTEM_PROMPT: str = """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task.
In each iteration, you will receive an Observation that includes the last  screenshots of a web browser and the current memory of the agent.
You have also information about the step that the agent is trying to achieve to solve the task.
Carefully analyze the visual information to identify what to do, then follow the guidelines to choose the following action.
You should detail your thought (i.e. reasoning steps) before taking the action.
Also detail in the notes field of the action the extracted information relevant to solve the task.
Once you have enough information in the notes to answer the task, return an answer action with the detailed answer in the notes field.
This will be evaluated by an evaluator and should match all the criteria or requirements of the task.

Guidelines:
- store in the notes all the relevant information to solve the task that fulfill the task criteria. Be precise
- Use both the task and the step information to decide what to do
- if you want to write in a text field and the text field already has text, designate the text field by the text it contains and its type
- If there is a cookies notice, always accept all the cookies first
- The observation is the screenshot of the current page and the memory of the agent.
- If you see relevant information on the screenshot to answer the task, add it to the notes field of the action.
- If there is no relevant information on the screenshot to answer the task, add an empty string to the notes field of the action.
- If you see buttons that allow to navigate directly to relevant information, like jump to ... or go to ... , use them to navigate faster.
- In the answer action, give as many details a possible relevant to answering the task.
- if you want to write, don't click before. Directly use the write action
- to write, identify the web element which is type and the text it already contains
- If you want to use a search bar, directly write text in the search bar
- Don't scroll too much. Don't scroll if the number of scrolls is greater than 3
- Don't scroll if you are at the end of the webpage
- Only refresh if you identify a rate limit problem
- If you are looking for a single flights, click on round-trip to select 'one way'
- Never try to login, enter email or password. If there is a need to login, then go back.
- If you are facing a captcha on a website, try to solve it.

- if you have enough information in the screenshot and in the notes to answer the task, return an answer action with the detailed answer in the notes field
- The current date is {timestamp}.

# <output_json_format>
# ```json
# {output_format}
# ```
# </output_json_format>

"""


def convert_image_to_base64_url(image: Image.Image, format: str = "JPEG") -> str:
    """Convert an image to a base64 URL.

    Args:
        image: PIL image.
        format: PIL image format (see https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html).

    Returns:
        Base64 URL.
    """
    mime_type = _get_mime_type_from_format(format=format)
    data = _encode_image_to_base64_string(image=image, format=format)
    return f"data:{mime_type};base64,{data}"


def _get_mime_type_from_format(format: str) -> str:
    """Get the MIME type associated to the PIL image format.

    Args:
        format: PIL image format (see https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html).

    Returns:
        MIME type associated to the PIL image format.
    """
    for extension, mime_type in types_map.items():
        pil_format = Image.registered_extensions().get(extension)
        if format == pil_format:
            return mime_type
    raise ValueError(f"Format {format} not supported")


def _encode_image_to_base64_string(
    image: Image.Image,
    format: str,
    jpeg_quality: int = 90,
) -> str:
    """Encodes an image to a base64 string.

    Args:
        image: PIL image.
        format: PIL image format (see https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html).
        jpeg_quality: JPEG compression quality.

    Returns:
        Base64 encoded string.
    """
    buffer = BytesIO()
    if format == "JPEG":
        image.convert("RGB").save(buffer, format=format, quality=jpeg_quality)
    else:
        image.save(buffer, format=format)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str


class ClickAbsoluteAction(BaseModel):
    """Click at absolute coordinates."""

    action: Literal["click_absolute"] = "click_absolute"
    x: int = Field(description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(description="The y coordinate, number of pixels from the top edge.")


class ClickElementAction(BaseModel):
    """Click at absolute coordinates of a web element with its description"""

    action: Literal["click_element"] = Field(
        description="Click at absolute coordinates of a web element"
    )
    element: str = Field(description="text description of the element")
    x: int = Field(description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(description="The y coordinate, number of pixels from the top edge.")

    def log(self):
        return f"I have clicked on the element '{self.element}' at absolute coordinates {self.x}, {self.y}"


class WriteElementAction(BaseModel):
    """Write content at absolute coordinates of a web element identified by its description, then press Enter."""

    action: Literal["write_element_abs"] = Field(
        description="Write content at absolute coordinates of a web page"
    )
    content: str = Field(description="Content to write")
    element: str = Field(description="Text description of the element")
    x: int = Field(description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(description="The y coordinate, number of pixels from the top edge.")

    def log(self):
        return f"I have written '{self.content}' in the element '{self.element}' at absolute coordinates {self.x}, {self.y}"


class ScrollAction(BaseModel):
    """Scroll action with no required element"""

    action: Literal["scroll"] = Field(
        description="Scroll the page or a specific element"
    )
    direction: Literal["down", "up", "left", "right"] = Field(
        description="The direction to scroll in"
    )

    def log(self):
        return f"I have scrolled {self.direction}"


class GoBackAction(BaseModel):
    """Action to navigate back in browser history"""

    action: Literal["go_back"] = Field(description="Navigate to the previous page")

    def log(self):
        return "I have gone back to the previous page"


class RefreshAction(BaseModel):
    """Action to refresh the current page"""

    action: Literal["refresh"] = Field(description="Refresh the current page")

    def log(self):
        return "I have refreshed the page"


class GotoAction(BaseModel):
    """Action to go to a particular URL"""

    action: Literal["goto"] = Field(description="Goto a particular URL")
    url: str = Field(description="A url starting with http:// or https://")

    def log(self):
        return f"I have navigated to the URL {self.url}"


class WaitAction(BaseModel):
    """Action to wait for a particular amount of time"""

    action: Literal["wait"] = Field(description="Wait for a particular amount of time")
    seconds: int = Field(
        default=2, ge=0, le=10, description="The number of seconds to wait"
    )

    def log(self):
        return f"I have waited for {self.seconds} seconds"


class RestartAction(BaseModel):
    """Restart the task from the beginning."""

    action: Literal["restart"] = "restart"

    def log(self):
        return "I have restarted the task from the beginning"


class AnswerAction(BaseModel):
    """Return a final answer to the task. This is the last action to call in an episode."""

    action: Literal["answer"] = "answer"
    content: str = Field(description="The answer content")

    def log(self):
        return f"I have answered the task with '{self.content}'"


ActionSpace = (
    ClickAbsoluteAction
    | WriteElementAction
    | ScrollAction
    | GoBackAction
    | RefreshAction
    | WaitAction
    | RestartAction
    | AnswerAction
    | GotoAction
)


class NavigationStep(BaseModel):
    note: str = Field(
        default="",
        description="Task-relevant information extracted from the previous observation. Keep empty if no new info.",
    )
    thought: str = Field(description="Reasoning about next steps (<4 lines)")
    action: ActionSpace = Field(description="Next action to take")

