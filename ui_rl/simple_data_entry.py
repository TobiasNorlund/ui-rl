from typing import List


class SimpleDataEntryTask:

    def __init__(self, rows: List):
        self.rows = rows

    def __str__(self):
        return f"SimpleDataEntryTask(rows={str(self.rows)})"

    def get_prompt(self):
        return f"""Your task is to submit data from a spreadsheet (seen on the left) into a form (seen on the right). Specifically, the following rows (as numbered in the left margin) from the spreadsheet are to be submitted: {", ".join(str(i) for i in self.rows)}.
Note: You may need to scroll to make the row visible in the sheet.
The form has to be submitted separately for each row. When the form has been submitted, return to the form to submit the next row. 
Submit a row by selecting each cell individually, copy its content by sending keys "ctrl+c", select the target form text input and paste using "ctrl+v".
Finally, click "Skicka" to submit the form, and continue with the next row. Only finish when all rows have been successfully submitted"""

    def get_state(self):
        return {"rows": self.rows}

    def get_pod_manifest(self, pod_name, session_id):
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "labels": {
                    "app": "simple-data-entry",
                    "session-id": session_id
                }
            },
            "spec": {
                "containers": [
                    {
                        "name": "session-container",
                        "image": "europe-north2-docker.pkg.dev/my-project-1726641910410/ui-verifiers/simple-data-entry:latest",
                        "imagePullPolicy": "Always",
                        "ports": [
                            {"containerPort": 8000},
                            {"containerPort": 5900}
                        ]
                    }
                ],
                "restartPolicy": "Never"
            }
        }