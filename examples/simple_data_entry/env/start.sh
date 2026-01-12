#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=:1

# Array to store background process PIDs
pids=()

# Function to cleanup background processes
cleanup() {
    echo "Received signal, cleaning up..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Wait a bit for processes to terminate gracefully
    sleep 1
    # Force kill any remaining processes
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start Xvfb
Xvfb :1 -ac -screen 0 1280x800x24 -dpi 96 -nolisten tcp &
pids+=($!)

# Run x11vnc with limited ulimit, due to a bug (slow when client connects) 
(ulimit -n 1024; x11vnc -display :1 -forever) &
pids+=($!)

# Start mutter
XDG_SESSION_TYPE=x11 mutter --replace --sm-disable &
pids+=($!)

sleep 2

uvicorn server:app --host 0.0.0.0 &
pids+=($!)

# Wait for all processes to finish
wait