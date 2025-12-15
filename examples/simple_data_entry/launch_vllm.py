import sys
import tempfile
import subprocess
import os

# --- Templates ---

NGINX_TEMPLATE = """
events {{}}
http {{
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
    image: vllm/vllm-openai:latest
    container_name: vllm-gpu-{id}
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      {extra_mount}
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
        return len(result.strip().split('\\n'))
    except Exception:
        print("⚠️  Could not detect GPUs. Defaulting to 1.")
        return 1

def generate_files(gpu_count: int, model_name: str, vllm_args):
    print(f"⚙️  Generating static config for {gpu_count} GPUs...")
    print(f"   Model: {model_name}")

    # 1. Nginx Config
    servers = [f"server vllm-gpu-{i}:8000 max_fails=1 fail_timeout=5s;" for i in range(gpu_count)]
    with open("nginx.conf", "w") as f:
        f.write(NGINX_TEMPLATE.format(server_list="\n        ".join(servers)))

    # 2. Docker Compose
    depends_on = [f"- vllm-gpu-{i}" for i in range(gpu_count)]
    services_yaml = ""
    
    for i in range(gpu_count):
        services_yaml += SERVICE_TEMPLATE.format(
            id=i,
            model_name=model_name,
            vllm_args=vllm_args
        )
        
    compose_content = (
        COMPOSE_HEADER.format(port=8000, depends_on_list="\n      ".join(depends_on)) +
        services_yaml + 
        COMPOSE_FOOTER
    )
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)


def launch(gpus: list[int], model_name: str, extra_mount: str | None, vllm_args: list[str]):
    # Write nginx.conf and docker compose
    with tempfile.TemporaryDirectory() as tmpdir:
        # Nginx config
        with open(os.path.join(tmpdir, "nginx.conf"), "w") as f:
            servers = [f"server vllm-gpu-{i}:8000 max_fails=1 fail_timeout=5s;" for i in gpus]
            f.write(NGINX_TEMPLATE.format(
                server_list="\n        ".join(servers),
            ))
        
        # Docker compose config
        extra_mount_str = f"- {extra_mount}" if extra_mount else ""
        services_yaml = ""
        for i in gpus:
            services_yaml += SERVICE_TEMPLATE.format(
                id=i,
                model_name=model_name,
                extra_mount=extra_mount_str,
                vllm_args=" ".join(vllm_args)
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
        subprocess.run(["docker", "compose", "-f", os.path.join(tmpdir, "docker-compose.yml"), "up"], env=os.environ.copy())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+")
    parser.add_argument("--model", default="ByteDance-Seed/UI-TARS-1.5-7B")
    parser.add_argument("--extra-mount")
    args, vllm_args = parser.parse_known_args()

    if args.gpus:
        gpus = int(args.gpus) if type(args.gpus) == str else [int(gpu) for gpu in args.gpus]
    else:
        gpus = list(range(get_gpu_count()))

    launch(
        gpus=gpus,
        model_name=args.model,
        extra_mount=args.extra_mount,
        vllm_args=vllm_args
    )