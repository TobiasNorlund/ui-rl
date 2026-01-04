import logging
from pathlib import Path
from queue import Queue
import re
from typing import List

from ui_rl.agent import run_cua_rollout
from ui_rl.models.uitars15.rollout import UITARS15_Rollout
from ui_rl.runner import FixedStrategy, NSuccessfulStrategy, RolloutResult, RolloutStrategy, RolloutWorker
from ui_rl.task import TaskSpec
from ui_rl.runtime.docker import DockerSessionRuntime


class SimpleDataEntryTaskSpec(TaskSpec[DockerSessionRuntime]):

    def __init__(self, rows: List[int]):
        self.rows = rows

    def __str__(self):
        return f"SimpleDataEntry(rows={str(self.rows)})"

    def get_task_instruction(self):
        return f"""Your task is to submit data from a spreadsheet (seen on the left) into a form (seen on the right). Specifically, the following rows (as numbered in the left margin) from the spreadsheet are to be submitted: {", ".join(str(i) for i in self.rows)}.
Note: You may need to scroll to make the row visible in the sheet.
The form has to be submitted separately for each row. When the form has been submitted, return to the form to submit the next row. 
Submit a row by selecting each cell individually, copy its content by sending keys "ctrl+c", select the target form text input and paste using "ctrl+v".
Finally, click "Skicka" to submit the form, and continue with the next row. Only finish when all rows have been successfully submitted"""

    def create_session(self, runtime: DockerSessionRuntime) -> str:
        return runtime.create_session(
            image="ui-rl/simple-data-entry:latest",
        )


class SimpleDataEntryRolloutWorker(RolloutWorker):
    _runtime = None

    def __init__(self, model_host: str, model_name: str, max_steps: int, output_dir: Path):
        self._model_host = model_host
        self._model_name = model_name
        self._max_steps = max_steps
        self._output_dir = output_dir

    @classmethod
    def init_worker(cls, log_queue: Queue):
        super().init_worker(log_queue)

        # Initialize docker runtime for this worker
        cls._runtime = DockerSessionRuntime(httpx_client=cls._httpx_client, session_timeout=60)

    def run(self, rollout_id: int, task_spec: SimpleDataEntryTaskSpec):
        rollout = UITARS15_Rollout(
            task_spec=task_spec,
            model_host=self._model_host,
            model_name=self._model_name,
            httpx_client=self._httpx_client,
            max_images_in_context=10,
            max_tokens=200,
            temperature=0.1
        )
        logging.info(f"Starting UITARS rollout for task: {task_spec}")
        try:
            run_cua_rollout(
                task_spec=task_spec,
                rollout=rollout,
                runtime=self._runtime,
                max_steps=self._max_steps,
            )
            logging.info(f"Rollout {rollout_id} completed")

            # Save rollout
            result = RolloutResult(rollout_id, task_spec, rollout.progress, None)
            file_name = f"row_{task_spec.rows[0]:03d}_success_{rollout_id:04d}.json" if rows_submitted_correctly(result) else f"row_{task_spec.rows[0]:03d}_fail_{rollout_id:04d}.json"
            rollout.save(self._output_dir / file_name)
            return result

        except Exception as e:
            logging.error(f"Rollout {rollout_id} was stopped due to an error: {e}")
            return RolloutResult(rollout_id, task_spec, rollout.progress, e)


def rows_submitted_correctly(result: RolloutResult) -> bool:
    """
    Checks if all the instructed spreadsheet rows were actually submitted
    Note: In the spreadsheet, the first data row starts at no 2, but the "submitted_row_indices"
          start from 0, so we need to subtract 2 from the instructed row when matching
    """
    assert isinstance(result.task_spec, SimpleDataEntryTaskSpec)
    return result.error is None and result.progress is not None and \
        set(result.progress["submitted_row_indices"]) == set([r-2 for r in result.task_spec.rows])


def parse_strategy(strategy: str) -> RolloutStrategy:
    """
    Creates a RolloutStrategy of single-row SimpleDataEntryTaskSpec tasks from string

    Examples:
     - "fixed(2-101)"             - FixedStrategy of SimpleDataEntryTaskSpec(rows=[i]) for i = 2..101 (inclusive)
     - "nsuccessful(2-101;1;2;3)" - NSuccessfulStrategy of SimpleDataEntryTaskSpec(rows=[i]) for i = 2..101 (inclusive)
                                    with min_successful=1, min_attempts=2, max_attempts=3
    """
    def _get_ids(ids: str):
        all_ids = list[int]()
        for id_group in ids.split(","):
            if "-" in id_group:
                start, stop = id_group.split("-")
                all_ids += list(range(int(start), int(stop)+1))
            else:
                all_ids.append(int(id_group))
        return all_ids

    match strategy:
        case s if (m := re.match(r"fixed\((?P<ids>\S+)\)", s)):
            ids = _get_ids(m.group("ids"))
            return FixedStrategy(tasks=[
                SimpleDataEntryTaskSpec(rows=[id])
                for id in ids
            ])
        case s if (m := re.match(r"nsuccessful\((?P<ids>\S+);(?P<min_successful>\d+);(?P<min_attempts>\d+);(?P<max_attempts>\d+)\)", s)):
            ids = _get_ids(m.group("ids"))
            return NSuccessfulStrategy(
                tasks=[
                    SimpleDataEntryTaskSpec(rows=[id])
                    for id in ids
                ], 
                min_successful=int(m.group("min_successful")),
                is_rollout_success=rows_submitted_correctly,
                min_attempts=int(m.group("min_attempts")),
                max_attempts=int(m.group("max_attempts"))
            )
        case _:
            raise ValueError("Invalid strategy")