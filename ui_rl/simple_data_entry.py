
def reward_fn(progress):
    # TODO: Improve
    return progress["num_correct_submissions"]


def simple_data_entry_pod_manifest(pod_name, session_id):
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