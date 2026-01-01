from contextlib import contextmanager
import shlex
import tempfile
import subprocess
import os

# --- Templates ---

NGINX_TEMPLATE = """
events {{}}
http {{
    # Allow large request bodies
    client_max_body_size 200M;

    upstream vllm_backend {{
        hash $http_x_routing_id consistent;
        {server_list}
    }}

    server {{
        listen 8000;

        location / {{
            proxy_pass http://vllm_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header Connection '';
            proxy_http_version 1.1;
            chunked_transfer_encoding off;
            proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
        }}
    }}
}}
"""

COMPOSE_HEADER = """
services:
  nginx:
    image: nginx:latest
    container_name: vllm-router
    ports:
      - "{port}:8000"
    volumes:
      - {dir}/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      {depends_on_list}
    networks:
      - vllm-network
"""

SERVICE_TEMPLATE = """
  vllm-gpu-{id}:
    image: {docker_image}
    container_name: vllm-gpu-{id}
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - VLLM_HTTP_TIMEOUT_KEEP_ALIVE=60
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
{mount}
    command: >
      {model_name}
      --port 8000
      {vllm_args}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['{id}']
              capabilities: [gpu]
    networks:
      - vllm-network
"""

COMPOSE_FOOTER = """
networks:
  vllm-network:
    driver: bridge
"""

# --- Logic ---

def get_gpu_count():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8"
        )
        return len(result.strip().split("\n"))
    except Exception:
        raise RuntimeError("Couldn't detect any GPU")


def remove_existing_containers(gpus: list[int]):
    """Remove existing vllm containers if they exist"""
    container_names = ["vllm-router"] + [f"vllm-gpu-{i}" for i in gpus]

    for container_name in container_names:
        try:
            # Check if container exists (running or stopped)
            result = subprocess.run(
                ["docker", "ps", "-a", "-q", "-f", f"name=^{container_name}$"],
                capture_output=True,
                encoding="utf-8"
            )
            if result.stdout.strip():
                print(f"🗑️  Removing existing container: {container_name}")
                subprocess.run(["docker", "rm", "-f", container_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to remove container {container_name}: {e}")


@contextmanager
def launch(
    gpus: list[int], 
    model_name: str, 
    mounts: list[str] | None, 
    docker_image: str = "vllm/vllm-openai:latest", 
    vllm_args: list[str] = [],
    detach: bool = False
):
    # Remove existing containers first
    remove_existing_containers(gpus)

    # Write nginx.conf and docker compose
    with tempfile.TemporaryDirectory() as tmpdir:
        # Nginx config
        with open(os.path.join(tmpdir, "nginx.conf"), "w") as f:
            servers = [f"server vllm-gpu-{i}:8000 max_fails=1 fail_timeout=5s;" for i in gpus]
            f.write(NGINX_TEMPLATE.format(
                server_list="\n        ".join(servers),
            ))
        
        # Docker compose config
        mount_str = "\n".join(f"      - {m}" for m in mounts) if mounts else ""
        services_yaml = ""
        for i in gpus:
            services_yaml += SERVICE_TEMPLATE.format(
                id=i,
                docker_image=docker_image,
                model_name=model_name,
                mount=mount_str,
                vllm_args=" ".join(shlex.quote(arg) for arg in vllm_args)
            )
        depends_on = [f"- vllm-gpu-{i}" for i in gpus]
        compose_content = (
            COMPOSE_HEADER.format(
                port=8000, 
                dir=tmpdir,
                depends_on_list="\n      ".join(depends_on), 
            ) +
            services_yaml + 
            COMPOSE_FOOTER
        )
        with open(os.path.join(tmpdir, "docker-compose.yml"), "w") as f:
            f.write(compose_content)
        
        # Launch
        proc = subprocess.Popen(["docker", "compose", "-f", os.path.join(tmpdir, "docker-compose.yml"), "up", "--detach" if detach else ""], env=os.environ.copy())

        yield proc

        # Cleanup
        subprocess.run(["docker", "compose", "-f", os.path.join(tmpdir, "docker-compose.yml"), "down"])


def main(gpus: list[int], model_name: str, docker_image: str, mounts: list[str] | None, vllm_args: list[str]):
    with launch(gpus, model_name, mounts, docker_image, vllm_args, detach=False) as proc:
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("Stopping...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+")
    parser.add_argument("--model", default="ByteDance-Seed/UI-TARS-1.5-7B")
    parser.add_argument("--image", default="vllm/vllm-openai:latest")
    parser.add_argument("--mount", nargs="+")
    args, vllm_args = parser.parse_known_args()

    if args.gpus:
        gpus = [int(gpu) for gpu in args.gpus]
    else:
        gpus = list(range(get_gpu_count()))

    main(
        gpus=gpus,
        model_name=args.model,
        mounts=args.mount,
        docker_image=args.image,
        vllm_args=vllm_args,
    )