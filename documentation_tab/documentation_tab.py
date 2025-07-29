"""Documentation tab for the pyMiceTracking Panel app.

This tab reads the top-level README.md and renders it as Markdown inside the
application so that users can view project documentation without leaving the
interface.
"""

from pathlib import Path
import panel as pn


def _load_readme() -> str:
    """Return README.md contents as a string.

    The file is expected to sit in the root directory one level above this
    module. If the file is missing, a helpful message is returned instead of
    raising an exception so the application can still start gracefully.
    """
    root_dir = Path(__file__).resolve().parent.parent
    readme_path = root_dir / "readme.md"
    try:
        return readme_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "# Documentation\n"
            "README.md was not found in the project root directory. "
            "Please ensure the file exists so the documentation tab can render it."
        )


def get_tab() -> pn.Column:
    """Return a Panel layout containing the project documentation."""
    md_pane = pn.pane.Markdown(_load_readme(), sizing_mode="stretch_width", max_width=900)
    return pn.Column(md_pane, sizing_mode="stretch_both")
