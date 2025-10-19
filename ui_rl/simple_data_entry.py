

class SimpleDataEntryTask:

    def get_prompt(self):
        return """Fill out the form on the right by copy pasting information from the FIRST data row (i.e. row no 2) in the Google Sheet on the left. 
Do this by selecting each cell individually, copy by sending keys "ctrl+c", select the target form text input, and paste using "ctrl+v".
Finally, click "Skicka" to submit the form."""

    def get_reward(self, progress):
        # TODO: Improve
        return progress["num_correct_submissions"]

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