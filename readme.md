# Modularized Project with Panel

## Folder Structure

```text
panel/
├── camera_tab/
│   └── camera_tab.py
├── tracking_tab/
│   └── tracking_tab.py
├── ethological_tab/
│   └── ethological_tab.py
├── irl_tab/
│   └── irl_tab.py
├── synthetic_tab/
│   └── synthetic_tab.py
├── experiments/
├── main.py
└── readme.md
```

- Each tab folder contains the logic, functions, and layout for its corresponding panel tab.
- The `experiments/` directory is a shared location used by all tabs for reading and writing files.
- `main.py` manages the overall interface and imports each tab as a mini-module.

## How to Run

```bash
panel serve main.py --show
```

## Environment Setup with uv

```bash
# Install uv (if not already installed)
curl -Ls https://astral.sh/uv/install.sh | sh

# Create a virtual environment inside the project
uv venv .venv

# Activate the environment (Linux/macOS)
source .venv/bin/activate

# Install required packages
uv pip install -r requirements.txt
```
