from typing import List
from ui_rl.task import TaskSpec
from ui_rl.runtime import CUASessionRuntime
from ui_rl.runtime.docker import DockerSessionRuntime


class SimpleDataEntryTaskSpec(TaskSpec):

    def __init__(self, rows: List[int]):
        self.rows = rows

    def __str__(self):
        return f"SimpleDataEntry(rows={str(self.rows)})"

    def __hash__(self):
        return hash(tuple(self.rows))

    def __eq__(self, other):
        if not isinstance(other, SimpleDataEntryTaskSpec):
            return False
        return self.rows == other.rows

    def get_task_instruction(self):
        return f"""Your task is to submit data from a spreadsheet (seen on the left) into a form (seen on the right). Specifically, the following rows (as numbered in the left margin) from the spreadsheet are to be submitted: {", ".join(str(i) for i in self.rows)}.
Note: You may need to scroll to make the row visible in the sheet.
The form has to be submitted separately for each row. When the form has been submitted, return to the form to submit the next row. 
Submit a row by selecting each cell individually, copy its content by sending keys "ctrl+c", select the target form text input and paste using "ctrl+v".
Finally, click "Skicka" to submit the form, and continue with the next row. Only finish when all rows have been successfully submitted"""

    def create_session(self, runtime: CUASessionRuntime) -> str:
        # Assume docker runtime
        assert isinstance(runtime, DockerSessionRuntime), "runtime must be a DockerSessionRuntime"
        return runtime.create_session(
            image="ui-rl/simple-data-entry:latest",
        )
