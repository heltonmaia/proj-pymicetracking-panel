import os
import shutil
from pathlib import Path
import panel as pn

# --- Limpeza automática de __pycache__ ---

def _clean_pycache(root: Path):
    """Remove todos os diretórios __pycache__ recursivamente a partir de *root*."""
    for dirpath, dirnames, _ in os.walk(root):
        if "__pycache__" in dirnames:
            cache_dir = Path(dirpath) / "__pycache__"
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass  # Ignorar erros de permissão

_clean_pycache(Path(__file__).parent)
# ----------------------------------------
from camera_tab.camera_tab import get_tab as get_camera_tab
from tracking_tab.tracking_tab import get_tab as get_tracking_tab
from ethological_tab.ethological_tab import get_tab as get_ethological_tab
from irl_tab.irl_tab import get_tab as get_irl_tab
from synthetic_tab.synthetic_tab import get_tab as get_synthetic_tab

# Configuração do Panel
pn.extension()

# Tabs
camera_tab = get_camera_tab()
tracking_tab = get_tracking_tab()
ethological_tab = get_ethological_tab()
irl_tab = get_irl_tab()
synthetic_tab = get_synthetic_tab()

control_tabs = pn.Tabs(
    ('📷 Camera & Recording', camera_tab),
    ('🔎 Animal Tracking', tracking_tab),
    ('🧬 Ethological Analysis', ethological_tab),
    ('🌐 IRL Analysis', irl_tab),
    ('🧪 Synthetic Data', synthetic_tab)
)

# Main layout
layout = pn.Column(
    pn.pane.HTML("<h1 style='text-align:center; color:#2E86C1; margin-bottom:20px;'>🐭 pyMiceTracking</h1>"),
    pn.Spacer(height=20),
    pn.Row(
        pn.Spacer(width=30),
        control_tabs,
        pn.Spacer(width=30),
        margin=(0, 0, 20, 0)
    ),
    width=900, margin=(20, 20)
)

# Marca como servable
layout.servable(title="pyMiceTracking")

# Para executar localmente
if __name__ == "__main__":
    layout.show(port=5007)