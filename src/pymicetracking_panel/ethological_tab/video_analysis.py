import base64
import threading
import time
from pathlib import Path
from typing import Callable

import panel as pn

from .tracking_visualizer import TrackingVisualizer


class VideoAnalysis:
    """Handles video analysis operations for ethological analysis"""

    def __init__(self, temp_dir: Path, project_root: Path) -> None:
        self.temp_dir = temp_dir
        self.project_root = project_root
        self.analysis_process = None
        self.is_analyzing = False
        self.progress_thread_running = False
        self.current_output_path = None

    def start_analysis(
        self,
        video_value: bytes,
        json_value: bytes,
        video_filename: str,
        json_filename: str,
        show_info: bool,
        show_heatmap: bool,
        progress_callback: Callable[[int], None],
        status_callback: Callable[[str, str], None],
    ) -> None:
        """Start video analysis in background thread"""
        if not video_value or not json_value:
            return

        # Start analysis mode
        self.is_analyzing = True

        status_callback(
            "**Status:** 🔄 Starting video analysis...\n\nInitializing analysis process...",
            "#fff3cd",
        )

        # Run analysis in background thread
        analysis_thread = threading.Thread(
            target=self._run_analysis,
            args=(
                video_value,
                json_value,
                video_filename,
                json_filename,
                show_info,
                show_heatmap,
                progress_callback,
                status_callback,
            ),
            daemon=True,
        )
        analysis_thread.start()

    def abort_analysis(self) -> None:
        """Abort the current analysis"""
        self.is_analyzing = False

    def _run_analysis(
        self,
        video_value: bytes,
        json_value: bytes,
        video_filename: str,
        json_filename: str,
        show_info: bool,
        show_heatmap: bool,
        progress_callback: Callable[[int], None],
        status_callback: Callable[[str, str], None],
    ) -> None:
        """Run the analysis in background thread"""
        try:
            video_stem = Path(video_filename).stem

            # Create temporary files for processing
            temp_video_path = self.project_root / f"temp_{video_filename}"
            temp_json_path = self.project_root / f"temp_{json_filename}"

            with open(temp_video_path, "wb") as f:
                f.write(video_value)

            with open(temp_json_path, "wb") as f:
                f.write(json_value)

            # Generate output filename - save to Downloads folder by default
            output_stem = video_stem + "_ethological_analysis"
            output_filename = f"{output_stem}{Path(video_filename).suffix}"

            # Save to temporary folder in ethological_tab
            output_path = self.temp_dir / output_filename

            progress_callback(10)
            status_callback(
                "**Status:** 🔄 Loading video and tracking data...\n\nPreparing files for processing...",
                "#fff3cd",
            )

            progress_callback(20)
            status_callback(
                "**Status:** 🔄 Creating visualizer...\n\nInitializing video processing engine...",
                "#fff3cd",
            )

            # Create visualizer
            visualizer = TrackingVisualizer(
                video_path=str(temp_video_path),
                json_path=str(temp_json_path),
                output_path=str(output_path),
                show_info=show_info,
                show_heatmap=show_heatmap,
            )

            progress_callback(30)
            status_callback(
                (
                    f"**Status:** 🔄 Processing video frames...\n\n"
                    f"Analyzing {visualizer.total_frames} frames with tracking data..."
                ),
                "#fff3cd",
            )

            # Process video with progress updates
            self._process_with_progress(visualizer, progress_callback, status_callback)

            if not self.is_analyzing:  # Check if aborted
                return

            # Cleanup
            visualizer.cleanup()

            progress_callback(100)

            # Small delay to ensure UI updates properly
            time.sleep(0.2)

            # Success
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            success_msg = (
                "**Status:** ✅ **Analysis Complete**"
                f"**File:** {output_path.name} ({file_size_mb:.1f} MB)"
                "**Location:** Temporary folder (ethological_tab/temp)"
                f"**Options:** Info Panel: {'✅' if show_info else '❌'} | Heatmap: {'✅' if show_heatmap else '❌'}"
                "Video processing completed successfully. Download link ready below."
            )

            status_callback(success_msg, "#d4edda")

            # Store output path for download
            self.current_output_path = output_path

        except ImportError as e:
            error_msg = (
                "**Status:** ❌ **Import Error**"
                f"Cannot import TrackingVisualizer: {str(e)}"
                "Make sure testVideo_json.py is in the ethological_tab directory."
            )
            status_callback(error_msg, "#f8d7da")

        except Exception as e:
            error_msg = (
                "**Status:** ❌ **Analysis Failed**"
                f"Error: {str(e)}"
                "Please check that the video and JSON files are valid."
            )
            status_callback(error_msg, "#f8d7da")

        finally:
            # Cleanup temp files
            for temp_file in ["temp_video_path", "temp_json_path"]:
                if temp_file in locals() and locals()[temp_file].exists():
                    try:
                        locals()[temp_file].unlink()
                    except:
                        pass

    def _process_with_progress(
        self,
        visualizer: TrackingVisualizer,
        progress_callback: Callable[[int], None],
        status_callback: Callable[[str, str], None],
    ) -> None:
        """Process video with progress updates"""
        try:
            self.progress_thread_running = True

            def update_progress() -> None:
                # Simulate progress updates while processing
                for i in range(30, 90, 10):
                    if (
                        not self.is_analyzing or not self.progress_thread_running
                    ):  # Check if aborted or should stop
                        return
                    time.sleep(1)  # Update every second
                    progress_callback(i)
                    status_callback(
                        (
                            f"**Status:** 🔄 Processing frames... ({i}%)\n\n"
                            f"Analyzing video with tracking data..."
                        ),
                        "#fff3cd",
                    )

            # Start progress update thread
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()

            # Process the video
            visualizer.process_video()

            # Stop progress thread and wait a moment for it to finish
            self.progress_thread_running = False
            time.sleep(0.5)

            # Complete progress
            if self.is_analyzing:  # Only if not aborted
                progress_callback(95)

        except Exception as e:
            raise e

    def download_file(self, status_callback: Callable[[str, str], None]) -> None:
        """Handle video file download"""
        if not self.current_output_path or not self.current_output_path.exists():
            return

        try:
            with open(self.current_output_path, "rb") as f:
                video_data = f.read()

            # Create a data URL for download
            b64_data = base64.b64encode(video_data).decode()

            # Create temporary download link and trigger it
            js_code = (
                "const link = document.createElement('a');"
                f"link.href = 'data:video/mp4;base64,{b64_data}';"
                f"link.download = '{self.current_output_path.name}';"
                "document.body.appendChild(link);"
                "link.click();"
                "document.body.removeChild(link);"
            )

            # Execute JavaScript to trigger download
            pn.io.push_notebook()
            pn.pane.HTML(f"<script>{js_code}</script>").servable()

        except Exception as e:
            status_callback(
                f"**Status:** ❌ Download Error\n\nCould not download file: {str(e)}",
                "#f8d7da",
            )
