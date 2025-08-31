# 🐭 pyMiceTracking Panel

A comprehensive mouse tracking application with Panel interface and YOLO-based computer vision for behavioral analysis.

## 🚀 Features

- **📷 Camera & Recording**: Live camera feed integration and recording
- **🔎 Animal Tracking**: YOLO-based mouse detection and tracking with GPU acceleration  
- **🧬 Ethological Analysis**: 
  - Video tracking analysis with heatmaps and info panels
  - Movement heatmap analysis with center of mass calculations
  - Individual plot generation and complete analysis panels
  - High-quality PNG/EPS export capabilities
- **🌐 IRL Analysis**: Real-world experiment integration
- **🧪 Synthetic Data**: Synthetic data generation tools
- **🛠️ Extra Tools**: Additional utilities and GPU testing

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

## 📦 Installation Extras

The package includes optional dependency groups for modular installation:

### GPU Support (CUDA acceleration)
```bash
uv pip install "pymicetracking-panel[gpu]"
```

### YOLO Only (minimal computer vision)
```bash  
uv pip install "pymicetracking-panel[yolo]"
```

### Visualization Only (analysis and plotting)
```bash
uv pip install "pymicetracking-panel[viz]"
```

### Development Tools (testing, linting, formatting)
```bash
uv pip install "pymicetracking-panel[dev]"

# Then use development commands
uv run black .              # Format code
uv run flake8 .             # Lint code  
uv run mypy .               # Type checking
uv run pytest              # Run tests
```

### All Extras
```bash
uv sync --all-extras
```

## 🔧 Dependencies

### Fixed Versions (Critical)
- **YOLO**: `ultralytics==8.3.102` (computer vision core)
- **PyTorch**: `torch==2.6.0`, `torchvision==0.21.0` (deep learning)
- **CUDA**: NVIDIA packages for GPU acceleration
- **OpenCV**: `opencv-python==4.11.0.86` (image processing)

### Flexible Versions (≥ minimum)
- **Scientific**: numpy, scipy, pandas, matplotlib, seaborn
- **Interface**: panel, bokeh and related packages  
- **Utilities**: tqdm, pyyaml, requests, shapely

See `pyproject.toml` for complete dependency specifications.

## 🧬 Ethological Analysis Features

The ethological analysis module provides comprehensive behavioral analysis tools:

### Video Tracking Analysis
- **Video + JSON input**: Process recorded videos with tracking data
- **Visualization options**: Info panels and movement heatmaps overlay
- **Real-time processing**: Background analysis with progress tracking

### Movement Heatmap Analysis  
- **Heatmap generation**: High-resolution movement density visualization
- **Center of mass analysis**: Distance calculations and movement patterns
- **Multiple visualizations**:
  - Distance from center over time
  - Movement velocity analysis  
  - Velocity distribution histograms
  - Activity classification (moving vs stationary)
  - Movement direction analysis (polar plots)
  - Cumulative distance tracking

### Configurable Parameters
- **Heatmap settings**: Resolution (20-100 bins), colormap selection, transparency
- **Movement analysis**: Threshold percentiles, histogram bins
- **Export options**: PNG/EPS formats at 300 DPI

### Analysis Modes
- **Complete Panel**: Single comprehensive figure with all analyses
- **Individual Plots**: Separate numbered figures for each analysis type

## 🖥️ GPU Support

This project includes CUDA dependencies for GPU acceleration:
- **Automatic fallback**: CPU processing if no CUDA GPU detected
- **Optimized performance**: Significant speedup with compatible hardware
- **Memory management**: Efficient handling of large video files

## 🧪 Development

### Code Quality Tools
```bash
# Format code
uv run black .

# Lint code  
uv run flake8 .

# Type checking
uv run mypy .

# Run tests
uv run pytest

# All quality checks
uv run black . && uv run flake8 . && uv run mypy . && uv run pytest
```



