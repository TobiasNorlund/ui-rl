from .cua_session import CUASession
from kubernetes import client, config
import uuid

# Load Kubernetes configuration once at module level
try:
    config.load_incluster_config()
except config.ConfigException:
    config.load_kube_config()

# Create Kubernetes API client once at module level
core_v1 = client.CoreV1Api()


def reward_fn(progress):
    # TODO: Improve
    return progress["num_correct_submissions"]


def create_session() -> CUASession:
    """
    Create a simple-data-entry pod, and wrap it in a CUA Session
    """
    
    # Generate a unique session ID
    session_id = str(uuid.uuid4())[:8]
    
    # Define the pod specification
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"simple-data-entry-session-{session_id}",
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
    
    # Create the pod
    try:
        pod_response = core_v1.create_namespaced_pod(
            namespace="default",
            body=pod_manifest
        )
        print(f"Pod created successfully: {pod_response.metadata.name}")
        print(f"Session ID: {session_id}")
        
        # Return a CUASession wrapper
        return CUASession(session_id=session_id)
        
    except Exception as e:
        print(f"Error creating pod: {e}")
        raise


if __name__ == "__main__":
    session = create_session()