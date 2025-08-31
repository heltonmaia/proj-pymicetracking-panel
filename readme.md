# 🐭 pyMiceTracking Panel

A comprehensive mouse tracking application with Panel interface and YOLO-based computer vision for behavioral analysis.

## 🚀 Features

- **Camera & Recording**: Live camera feed integration and recording
- **Animal Tracking**: YOLO-based mouse detection and tracking with GPU acceleration  
- **Ethological Analysis**: 
  - Video tracking analysis with heatmaps and info panels
  - Movement heatmap analysis with center of mass calculations
  - Individual plot generation and complete analysis panels
  - High-quality PNG/EPS export capabilities
- **IRL Analysis**: Real-world experiment integration
- **Synthetic Data**: Synthetic data generation tools
- **Extra Tools**: Additional utilities and GPU testing

## 📋 Prerequisites

- **Python**: ≥3.11
- **Package Manager**: [UV](https://docs.astral.sh/uv/) (recommended) or pip
- **GPU**: CUDA-compatible GPU recommended for optimal performance
- **System**: Linux/Windows/macOS

## 🏗️ Project Structure

```
proj-pymicetracking-panel/
├── 📄 pyproject.toml           # Project configuration (optimized)
├── 📄 readme.md               # This documentation
├── 📄 .gitignore             # Git ignored files
│
├── 📁 src/                   # Source code (professional layout)
│   └── 📁 pymicetracking_panel/
│       ├── 📄 __init__.py        # Package initialization  
│       ├── 📄 main.py           # Main application entry point
│       │
│       ├── 📁 camera_tab/       # Camera & Recording module
│       ├── 📁 tracking_tab/     # Animal Tracking module
│       │   ├── 📁 processing/   # Detection and tracking logic
│       │   ├── 📁 models/       # YOLO model files
│       │   └── 📁 temp/         # Temporary processing files
│       │
│       ├── 📁 ethological_tab/  # Ethological Analysis module  
│       │   ├── 📄 ethological_tab.py   # Main analysis interface
│       │   ├── 📄 testVideo_json.py    # Video processing utilities
│       │   └── 📁 temp/                # Analysis output files
│       │
│       ├── 📁 irl_tab/          # IRL Analysis module
│       ├── 📁 synthetic_tab/    # Synthetic Data module  
│       └── 📁 extra_tools_tab/  # Extra Tools module
│
├── 📁 tests/                 # Unit tests
├── 📁 experiments/          # Experiment data (gitignored)
└── 📁 models/               # Additional models
```

## ⚡ Quick Start

### Using UV (Recommended)

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone https://github.com/heltonmaia/proj-pymicetracking-panel.git
   cd proj-pymicetracking-panel
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Run the application**:
   ```bash
   uv run panel serve src/pymicetracking_panel/main.py --show
   ```

The application will open automatically in your browser at `http://localhost:5006/main`

### Alternative Installation Methods

#### Development Mode
```bash
# Install in editable mode
uv pip install -e .

# Or with extras
uv pip install -e ".[dev,gpu,viz]"

# Then run
uv run pymicetracking
```

#### Build and Install
```bash
# Build the package
uv build

# Install the wheel
uv pip install dist/pymicetracking_panel-0.1.0-py3-none-any.whl

# Run via command
pymicetracking
```

#### Traditional pip
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
panel serve src/pymicetracking_panel/main.py --show
```

## Installation Extras

Optional dependency groups for modular installation:

```bash
# GPU acceleration (CUDA support)
uv pip install "pymicetracking-panel[gpu]"

# Computer vision only (minimal YOLO)
uv pip install "pymicetracking-panel[yolo]"

# Analysis and plotting tools
uv pip install "pymicetracking-panel[viz]"

# Development tools (testing, linting, formatting)
uv pip install "pymicetracking-panel[dev]"

# Install all extras
uv sync --all-extras
```



