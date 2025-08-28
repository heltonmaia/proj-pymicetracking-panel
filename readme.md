# pyMiceTracking Panel

A mouse tracking application with Panel interface and YOLO-based computer vision for behavioral analysis.

## Features

- **Camera Tab**: Live camera feed integration
- **Tracking Tab**: YOLO-based mouse detection and tracking
- **Ethological Tab**: Behavioral analysis tools
- **IRL Tab**: Real-world experiment integration
- **Synthetic Tab**: Synthetic data generation
- **Playback Tab**: Video playback and analysis
- **Documentation Tab**: User guides and help

## Prerequisites

- Python ≥3.11
- [UV](https://docs.astral.sh/uv/) package manager
- CUDA-compatible GPU (recommended for optimal performance)

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd proj-pymicetracking-panel
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Run the application:**
   ```bash
   uv run panel serve main.py --show
   ```

The application will open in your default browser at `http://localhost:5006/main`

## Project Structure

```text
proj-pymicetracking-panel/
├── camera_tab/
│   └── camera_tab.py
├── tracking_tab/
│   ├── tracking_tab.py
│   ├── processing/
│   │   ├── detection.py
│   │   └── tracking.py
│   ├── models/
│   │   └── yolo_model.pt
│   └── temp/
├── ethological_tab/
│   └── ethological_tab.py
├── irl_tab/
│   └── irl_tab.py
├── synthetic_tab/
│   └── synthetic_tab.py
├── playback_tab/
│   └── playback_tab.py
├── documentation_tab/
│   └── documentation_tab.py
├── experiments/          # Shared data location (gitignored)
├── main.py              # Main application entry point
├── pyproject.toml       # Project dependencies and configuration
└── readme.md
```

## Development Setup

### Using UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project
git clone <repository-url>
cd proj-pymicetracking-panel

# Install all dependencies (including dev dependencies)
uv sync --all-extras

# Run the application
uv run panel serve main.py --show

# Run development tools
uv run black .              # Format code
uv run flake8 .             # Lint code  
uv run mypy .               # Type checking
uv run pytest              # Run tests
```

### Alternative Setup (Traditional)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Run application
panel serve main.py --show
```

## GPU Support

This project includes CUDA dependencies for GPU acceleration. If you don't have a CUDA-compatible GPU, the application will fall back to CPU processing (slower performance).

## Dependencies

Key dependencies include:
- **Panel**: Web app framework
- **OpenCV**: Computer vision
- **Ultralytics YOLO**: Object detection
- **PyTorch**: Deep learning framework
- **NumPy/Pandas**: Data processing

See `pyproject.toml` for complete dependency list.

## License

MIT License
