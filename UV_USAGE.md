# Using UV with UI-RL

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

## Quick Start

### First Time Setup

```bash
# Install dependencies and create virtual environment
uv sync

# Or install with dev dependencies
uv sync --extra dev

# On Linux with CUDA (for quantization support)
uv sync --extra quantization
```

### Running Scripts

```bash
# Run any script with uv
uv run python scripts/test_vm_connection.py --vm-url http://VM_IP:8000

# Run with multiple runners
uv run python scripts/test_vm_connection.py --vm-url http://VM_IP:8000 --num-runners 3

# Run training script
uv run python scripts/train.py --config config/default_config.yaml
```


### Development Workflow

```bash
# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Update all dependencies
uv sync --upgrade

# Remove a dependency
uv remove package-name

# Lock dependencies without installing
uv lock

# Show installed packages
uv pip list
```

### Using the Virtual Environment Directly

```bash
# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Now you can run commands directly
python scripts/test_vm_connection.py --vm-url http://VM_IP:8000

# Deactivate when done
deactivate
```

## Project Structure

- `pyproject.toml` - Project metadata and dependencies
- `.venv/` - Virtual environment (created by uv)
- `uv.lock` - Locked dependency versions (commit this!)
- `src/` - Source code
- `scripts/` - Executable scripts
- `tests/` - Test files

## Optional Dependencies

- `dev` - Development tools (pytest, black, mypy, etc.)
- `quantization` - GPU quantization support (Linux only, includes bitsandbytes)

Install with: `uv sync --extra dev --extra quantization`

## Troubleshooting

### Import errors
If you get import errors, rebuild the package:
```bash
uv sync --reinstall
```

### Platform-specific dependencies
Some packages like `bitsandbytes` only work on Linux. They're marked as optional in the `quantization` extra.

### Clearing cache
```bash
uv cache clean
```
