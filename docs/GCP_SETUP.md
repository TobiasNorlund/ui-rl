# GCP Setup Guide for UI-RL Training

This guide explains how to set up UI-RL training with ui-verifiers running on Google Cloud Platform (GCP) VMs.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        GCP Project                          │
│                                                             │
│  ┌──────────────────┐                                      │
│  │  Trainer VM      │                                      │
│  │  (GPU Instance)  │                                      │
│  │                  │                                      │
│  │  ┌────────────┐  │                                      │
│  │  │   VLM      │  │                                      │
│  │  │ (Qwen2-VL) │  │                                      │
│  │  └────────────┘  │                                      │
│  │                  │                                      │
│  │  ┌────────────┐  │                                      │
│  │  │  Trainer   │──┼────┐                                │
│  │  └────────────┘  │    │                                │
│  │                  │    │ HTTP Requests                   │
│  │  ┌────────────┐  │    │                                │
│  │  │  Actors    │──┼────┤                                │
│  │  │  (x8)      │  │    │                                │
│  │  └────────────┘  │    │                                │
│  └──────────────────┘    │                                │
│                          │                                │
│                          ▼                                │
│  ┌─────────────────────────────────────────────┐          │
│  │         UI Verifier VMs (x8)                │          │
│  │                                             │          │
│  │  ┌──────────────────┐  ┌──────────────────┐ │          │
│  │  │ ui-verifier-vm-1 │  │ ui-verifier-vm-2 │ │          │
│  │  │                  │  │                  │ │          │
│  │  │ Port 8000        │  │ Port 8000        │ │  ...     │
│  │  │ (FastAPI)        │  │ (FastAPI)        │ │          │
│  │  │                  │  │                  │ │          │
│  │  │ Xvfb + GNOME     │  │ Xvfb + GNOME     │ │          │
│  │  └──────────────────┘  └──────────────────┘ │          │
│  └─────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- GCP Project with billing enabled
- `gcloud` CLI installed and authenticated
- Terraform (optional, for infrastructure as code)
- Access to create VMs, networks, and firewall rules

## Step 1: Network Setup

### Create VPC Network (if not using default)

```bash
# Create custom VPC
gcloud compute networks create ui-rl-network \
    --subnet-mode=auto \
    --project=YOUR_PROJECT_ID

# Create firewall rule for ui-verifiers API
gcloud compute firewall-rules create allow-ui-verifiers \
    --network=ui-rl-network \
    --allow=tcp:8000 \
    --source-ranges=10.128.0.0/9 \
    --description="Allow ui-verifiers API access within VPC" \
    --project=YOUR_PROJECT_ID
```

### Create Firewall Rule for Internal Communication

```bash
# Allow internal communication between trainer and ui-verifiers
gcloud compute firewall-rules create allow-internal-ui-rl \
    --network=ui-rl-network \
    --allow=tcp:8000,tcp:22 \
    --source-tags=ui-rl-trainer \
    --target-tags=ui-verifier \
    --project=YOUR_PROJECT_ID
```

## Step 2: Create UI Verifier VMs

### VM Specifications

- **Machine Type**: `n1-standard-4` (4 vCPUs, 15 GB RAM)
- **OS**: Ubuntu 22.04 LTS
- **Required Software**: Xvfb, gnome-session, Python 3.10+

### Create VM Script

```bash
#!/bin/bash
# create_ui_verifier_vms.sh

PROJECT_ID="your-project-id"
ZONE="us-central1-a"
NETWORK="ui-rl-network"
NUM_VMS=8

for i in $(seq 1 $NUM_VMS); do
    VM_NAME="ui-verifier-vm-${i}"

    echo "Creating ${VM_NAME}..."

    gcloud compute instances create ${VM_NAME} \
        --project=${PROJECT_ID} \
        --zone=${ZONE} \
        --machine-type=n1-standard-4 \
        --network-interface=network-tier=PREMIUM,subnet=default \
        --tags=ui-verifier \
        --metadata=startup-script='#!/bin/bash
# Install dependencies
apt-get update
apt-get install -y xvfb gnome-session python3-pip python3-venv git

# Clone ui-verifiers
cd /opt
git clone https://github.com/yourusername/ui-verifiers.git
cd ui-verifiers

# Install ui-verifiers
python3 -m venv venv
source venv/bin/activate
pip install uv
uv pip install -e .

# Create systemd service
cat > /etc/systemd/system/ui-verifiers.service <<EOF
[Unit]
Description=UI Verifiers Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ui-verifiers
Environment="PATH=/opt/ui-verifiers/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/opt/ui-verifiers/venv/bin/uvicorn ui_verifiers.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable ui-verifiers.service
systemctl start ui-verifiers.service
' \
        --maintenance-policy=MIGRATE \
        --provisioning-model=STANDARD \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-standard \
        --boot-disk-device-name=${VM_NAME}

    echo "Created ${VM_NAME}"
done

echo "All UI verifier VMs created!"
```

### Manual VM Setup (Alternative)

If you prefer manual setup:

```bash
# SSH into VM
gcloud compute ssh ui-verifier-vm-1 --zone=us-central1-a

# Install dependencies
sudo apt-get update
sudo apt-get install -y xvfb gnome-session python3-pip python3-venv git

# Clone and install ui-verifiers
cd /opt
sudo git clone https://github.com/yourusername/ui-verifiers.git
cd ui-verifiers
python3 -m venv venv
source venv/bin/activate
pip install uv
uv pip install -e .

# Start server (for testing)
uvicorn ui_verifiers.server:app --host 0.0.0.0 --port 8000

# Or use systemd service (see script above)
```

## Step 3: Create Trainer VM

### VM Specifications

- **Machine Type**: `n1-standard-8` (8 vCPUs, 30 GB RAM)
- **GPU**: NVIDIA Tesla T4 or V100
- **OS**: Ubuntu 22.04 LTS with CUDA drivers

### Create Trainer VM

```bash
#!/bin/bash
# create_trainer_vm.sh

PROJECT_ID="your-project-id"
ZONE="us-central1-a"

gcloud compute instances create ui-rl-trainer \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --tags=ui-rl-trainer \
    --metadata=startup-script='#!/bin/bash
# Install CUDA and cuDNN
apt-get update
apt-get install -y cuda-drivers cuda-toolkit-12-1

# Install Python and dependencies
apt-get install -y python3-pip python3-venv git

# Clone ui-rl repository
cd /home/ubuntu
git clone https://github.com/yourusername/ui-rl.git
cd ui-rl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
' \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --boot-disk-device-name=ui-rl-trainer
```

## Step 4: Configure Training

### Update Configuration File

SSH into trainer VM and edit the configuration:

```bash
gcloud compute ssh ui-rl-trainer --zone=us-central1-a

cd /home/ubuntu/ui-rl
source venv/bin/activate

# Edit config file
nano config/gcp_training.yaml
```

Update the `ui_env_urls` to match your VM names:

```yaml
environment:
  ui_env_urls:
    - "http://ui-verifier-vm-1:8000"
    - "http://ui-verifier-vm-2:8000"
    - "http://ui-verifier-vm-3:8000"
    - "http://ui-verifier-vm-4:8000"
    - "http://ui-verifier-vm-5:8000"
    - "http://ui-verifier-vm-6:8000"
    - "http://ui-verifier-vm-7:8000"
    - "http://ui-verifier-vm-8:8000"
```

### Verify Network Connectivity

Test that the trainer can reach ui-verifiers:

```bash
# From trainer VM
for i in {1..8}; do
    echo "Testing ui-verifier-vm-${i}..."
    curl http://ui-verifier-vm-${i}:8000/
done
```

You should see: `UI Verifiers Server v0.1.0`

## Step 5: Start Training

### Run Training Script

```bash
cd /home/ubuntu/ui-rl
source venv/bin/activate

# Start training
python scripts/train.py --config config/gcp_training.yaml
```

### Monitor Training (in separate terminal)

```bash
# SSH into trainer VM
gcloud compute ssh ui-rl-trainer --zone=us-central1-a

# Monitor logs
tail -f experiments/gcp_training/logs/train.log

# Or use tensorboard (if configured)
tensorboard --logdir experiments/gcp_training/logs
```

## Step 6: Monitor Resources

### Check VM Status

```bash
# List all VMs
gcloud compute instances list --filter="name~'ui-.*'"

# Check specific VM
gcloud compute instances describe ui-verifier-vm-1 --zone=us-central1-a
```

### Check ui-verifiers Status

```bash
# Check sessions on a VM
curl http://ui-verifier-vm-1:8000/status

# SSH and check logs
gcloud compute ssh ui-verifier-vm-1 --zone=us-central1-a
sudo journalctl -u ui-verifiers -f
```

## Cost Optimization

### Use Preemptible VMs for ui-verifiers

Preemptible VMs are 60-91% cheaper:

```bash
# Add these flags when creating ui-verifier VMs
--preemptible \
--provisioning-model=SPOT \
--instance-termination-action=STOP
```

### Auto-shutdown During Idle

Create a script to stop VMs when not training:

```bash
#!/bin/bash
# stop_training_vms.sh

PROJECT_ID="your-project-id"
ZONE="us-central1-a"

# Stop trainer VM
gcloud compute instances stop ui-rl-trainer --zone=${ZONE}

# Stop ui-verifier VMs
for i in {1..8}; do
    gcloud compute instances stop ui-verifier-vm-${i} --zone=${ZONE}
done
```

### Resume Training

```bash
#!/bin/bash
# start_training_vms.sh

PROJECT_ID="your-project-id"
ZONE="us-central1-a"

# Start ui-verifier VMs first
for i in {1..8}; do
    gcloud compute instances start ui-verifier-vm-${i} --zone=${ZONE}
done

# Wait for VMs to start
sleep 30

# Start trainer VM
gcloud compute instances start ui-rl-trainer --zone=${ZONE}
```

## Troubleshooting

### Cannot Connect to ui-verifiers

1. **Check firewall rules**:
   ```bash
   gcloud compute firewall-rules list --filter="name~'ui-.*'"
   ```

2. **Verify ui-verifiers is running**:
   ```bash
   gcloud compute ssh ui-verifier-vm-1 --zone=us-central1-a
   sudo systemctl status ui-verifiers
   ```

3. **Check service logs**:
   ```bash
   sudo journalctl -u ui-verifiers -n 100
   ```

### Session Creation Fails

1. **Check Xvfb and GNOME**:
   ```bash
   ps aux | grep -E "Xvfb|gnome-session"
   ```

2. **Test session creation manually**:
   ```bash
   curl -X POST "http://ui-verifier-vm-1:8000/session?type=simple_data_entry&n=1"
   ```

### Out of Memory

- Increase VM memory: Use `n1-standard-8` instead of `n1-standard-4`
- Reduce number of concurrent sessions
- Monitor memory usage: `htop` or `free -h`

### GPU Not Available

1. **Check GPU is attached**:
   ```bash
   nvidia-smi
   ```

2. **Install CUDA drivers**:
   ```bash
   sudo apt-get install -y nvidia-driver-525 cuda-drivers
   ```

3. **Verify PyTorch sees GPU**:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   ```

## Cleanup

### Delete All Resources

```bash
#!/bin/bash
# cleanup.sh

PROJECT_ID="your-project-id"
ZONE="us-central1-a"

# Delete trainer VM
gcloud compute instances delete ui-rl-trainer --zone=${ZONE} --quiet

# Delete ui-verifier VMs
for i in {1..8}; do
    gcloud compute instances delete ui-verifier-vm-${i} --zone=${ZONE} --quiet
done

# Delete firewall rules (optional)
gcloud compute firewall-rules delete allow-ui-verifiers --quiet
gcloud compute firewall-rules delete allow-internal-ui-rl --quiet

echo "Cleanup complete!"
```