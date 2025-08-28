import panel as pn
import os
import json
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
import cv2 as cv
import numpy as np

class EthologicalTab:
    def __init__(self):
        # Get paths
        self.project_root = Path(__file__).parent.parent
        self.experiments_dir = self.project_root / 'experiments'
        self.temp_dir = Path(__file__).parent / 'temp'
        
        # Clean and create temp directory
        self._cleanup_temp_files()
        self.temp_dir.mkdir(exist_ok=True)
        
        # Video Analysis components
        self.video_input = pn.widgets.FileInput(
            accept='.mp4,.avi,.mov,.mkv',
            name='📹 Select Video File:',
            width=400,
            multiple=False
        )
        
        self.json_input = pn.widgets.FileInput(
            accept='.json',
            name='📄 Select JSON File:',
            width=400,
            multiple=False
        )
        
        
        self.show_info_panel = pn.widgets.Checkbox(
            name='Show Info Panel on Video',
            value=False,
            width=200
        )
        
        self.show_heatmap = pn.widgets.Checkbox(
            name='Show Heatmap on Video',
            value=False,
            width=200
        )
        
        self.start_analysis_button = pn.widgets.Button(
            name='🚀 Start Analysis',
            button_type='success',
            width=150,
            height=40,
            disabled=True
        )
        
        self.abort_button = pn.widgets.Button(
            name='❌ Abort Analysis',
            button_type='danger',
            width=150,
            height=40,
            disabled=True,
            visible=False
        )
        
        # Progress and results
        self.analysis_progress = pn.indicators.Progress(
            name='Analysis Progress',
            value=0,
            max=100,
            width=400,
            visible=False
        )
        
        # Unified status panel
        self.unified_status = pn.pane.Markdown(
            "**Status:** Ready\n\nSelect video and JSON files to start analysis.",
            styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '5px'},
            width=500,
            height=200
        )
        
        # Download button component
        self.download_button = pn.widgets.Button(
            name='📥 Download Result',
            button_type='success',
            width=150,
            height=40,
            disabled=True,
            visible=False
        )
        
        # Store output path for download
        self.current_output_path = None
        
        # Analysis control
        self.analysis_process = None
        self.is_analyzing = False
        self.progress_thread_running = False
        
        # Connect events
        self.video_input.param.watch(self._on_file_selected, 'value')
        self.video_input.param.watch(self._on_file_selected, 'filename')
        self.json_input.param.watch(self._on_file_selected, 'value')
        self.json_input.param.watch(self._on_file_selected, 'filename')
        self.start_analysis_button.on_click(self._start_analysis)
        self.abort_button.on_click(self._abort_analysis)
        self.download_button.on_click(self._download_file)
    
    def _cleanup_temp_files(self):
        """Clean up temporary files from previous sessions"""
        temp_dir = Path(__file__).parent / 'temp'
        if temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print(f"[CLEANUP] Removed temporary files from: {temp_dir}")
            except Exception as e:
                print(f"[CLEANUP] Warning: Could not clean temp directory: {e}")
    
    def _on_file_selected(self, event):
        """Handle file selection for both video and JSON"""
        self._update_files_status()
    
    def _update_files_status(self):
        """Update status based on selected files"""
        has_video = self.video_input.value and len(self.video_input.value) > 0
        has_json = self.json_input.value and len(self.json_input.value) > 0
        
        if not has_video and not has_json:
            self._update_unified_status("**Status:** Ready\n\nSelect video and JSON files to start analysis.", '#f8f9fa')
            self.start_analysis_button.disabled = True
            return
        
        if not has_video:
            self._update_unified_status("**Status:** ❌ Video file required\n\nPlease select a video file to continue.", '#f8d7da')
            self.start_analysis_button.disabled = True
            return
            
        if not has_json:
            self._update_unified_status("**Status:** ❌ JSON file required\n\nPlease select a JSON tracking file to continue.", '#f8d7da')
            self.start_analysis_button.disabled = True
            return
        
        # Both files selected - validate JSON
        try:
            # Create temporary JSON file to validate
            temp_json_path = self.project_root / 'temp_json_for_validation.json'
            with open(temp_json_path, 'wb') as f:
                f.write(self.json_input.value)
            
            if self._validate_json_file(temp_json_path):
                video_filename = getattr(self.video_input, 'filename', 'video_file.mp4')
                json_filename = getattr(self.json_input, 'filename', 'data.json')
                
                video_size_mb = len(self.video_input.value) / 1024 / 1024
                status_msg = f"**Status:** ✅ Ready to analyze\n\n**Video:** {video_filename} ({video_size_mb:.1f} MB)\n**JSON:** {json_filename}\n\nClick 'Start Analysis' to begin processing."
                self._update_unified_status(status_msg, '#d1ecf1')
                
                self.start_analysis_button.disabled = False
            else:
                self._update_unified_status("**Status:** ❌ Invalid JSON format\n\nThe JSON file doesn't contain valid tracking data. Please select a proper tracking JSON file.", '#f8d7da')
                self.start_analysis_button.disabled = True
            
            # Cleanup temp file
            if temp_json_path.exists():
                temp_json_path.unlink()
                
        except Exception as e:
            self._update_unified_status(f"**Status:** ❌ Error validating JSON\n\n{str(e)}\n\nPlease check your JSON file format.", '#f8d7da')
            self.start_analysis_button.disabled = True
    
    def _update_unified_status(self, message: str, background_color: str):
        """Update unified status panel"""
        self.unified_status.object = message
        self.unified_status.styles = {'background': background_color, 'padding': '15px', 'border-radius': '5px'}
    
    def _find_json_by_name(self, video_stem: str) -> Optional[Path]:
        """Find corresponding JSON file by video name in experiments directory"""
        if not self.experiments_dir.exists():
            return None
        
        # Pattern 1: tracking_data_<video_name>_<timestamp>.json
        for json_file in self.experiments_dir.rglob(f"tracking_data_{video_stem}*.json"):
            return json_file
        
        # Pattern 2: <video_name>.json
        for json_file in self.experiments_dir.rglob(f"{video_stem}.json"):
            return json_file
        
        # Pattern 3: Look for any JSON file with video stem in name
        for json_file in self.experiments_dir.rglob("*.json"):
            if video_stem in json_file.stem:
                return json_file
        
        return None
    
    def _validate_json_file(self, json_path: Path) -> bool:
        """Validate that JSON file contains tracking data"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check for required structure
            if not isinstance(data, dict):
                return False
            
            # Check for tracking_data field
            tracking_data = data.get("tracking_data", [])
            if not isinstance(tracking_data, list):
                return False
            
            # If there's data, check first frame structure
            if len(tracking_data) > 0:
                first_frame = tracking_data[0]
                required_fields = ["frame_number", "centroid_x", "centroid_y"]
                for field in required_fields:
                    if field not in first_frame:
                        return False
            
            return True
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return False
    
    def _start_analysis(self, event):
        """Start ethological analysis"""
        if not self.video_input.value or not self.json_input.value:
            return
        
        # Start analysis mode
        self.is_analyzing = True
        self.start_analysis_button.disabled = True
        self.abort_button.visible = True
        self.abort_button.disabled = False
        self.analysis_progress.visible = True
        self.analysis_progress.value = 0
        self.download_button.visible = False
        
        self._update_unified_status("**Status:** 🔄 Starting video analysis...\n\nInitializing analysis process...", '#fff3cd')
        
        # Run analysis in background thread
        analysis_thread = threading.Thread(target=self._run_analysis, daemon=True)
        analysis_thread.start()
    
    def _run_analysis(self):
        """Run the analysis in background thread"""
        try:
            # Get filenames
            video_filename = getattr(self.video_input, 'filename', 'video_file.mp4')
            json_filename = getattr(self.json_input, 'filename', 'data.json')
            video_stem = Path(video_filename).stem
            
            # Create temporary files for processing
            temp_video_path = self.project_root / f"temp_{video_filename}"
            temp_json_path = self.project_root / f"temp_{json_filename}"
            
            with open(temp_video_path, 'wb') as f:
                f.write(self.video_input.value)
            
            with open(temp_json_path, 'wb') as f:
                f.write(self.json_input.value)
            
            # Generate output filename - save to Downloads folder by default
            output_stem = video_stem + "_ethological_analysis"
            output_filename = f"{output_stem}{Path(video_filename).suffix}"
            
            # Save to temporary folder in ethological_tab
            output_path = self.temp_dir / output_filename
            
            self.analysis_progress.value = 10
            self._update_unified_status("**Status:** 🔄 Loading video and tracking data...\n\nPreparing files for processing...", '#fff3cd')
            
            # Import and use the TrackingVisualizer from testVideo_json.py
            from .testVideo_json import TrackingVisualizer
            
            self.analysis_progress.value = 20
            self._update_unified_status("**Status:** 🔄 Creating visualizer...\n\nInitializing video processing engine...", '#fff3cd')
            
            # Create visualizer
            visualizer = TrackingVisualizer(
                video_path=str(temp_video_path),
                json_path=str(temp_json_path),
                output_path=str(output_path),
                show_info=self.show_info_panel.value,
                show_heatmap=self.show_heatmap.value
            )
            
            self.analysis_progress.value = 30
            self._update_unified_status(f"**Status:** 🔄 Processing video frames...\n\nAnalyzing {visualizer.total_frames} frames with tracking data...", '#fff3cd')
            
            # Process video with progress updates
            self._process_with_progress(visualizer)
            
            if not self.is_analyzing:  # Check if aborted
                return
            
            # Cleanup
            visualizer.cleanup()
            
            self.analysis_progress.value = 100
            
            # Small delay to ensure UI updates properly
            time.sleep(0.2)
            
            # Success
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            success_msg = f"""**Status:** ✅ **Analysis Complete**

**File:** {output_path.name} ({file_size_mb:.1f} MB)
**Location:** Temporary folder (ethological_tab/temp)
**Options:** Info Panel: {'✅' if self.show_info_panel.value else '❌'} | Heatmap: {'✅' if self.show_heatmap.value else '❌'}

Video processing completed successfully. Download link ready below."""
            
            self._update_unified_status(success_msg, '#d4edda')
            
            # Enable download button
            self.current_output_path = output_path
            self.download_button.disabled = False
            self.download_button.visible = True
            
        except ImportError as e:
            error_msg = f"""**Status:** ❌ **Import Error**

Cannot import TrackingVisualizer: {str(e)}

Make sure testVideo_json.py is in the ethological_tab directory."""
            self._update_unified_status(error_msg, '#f8d7da')
            
        except Exception as e:
            error_msg = f"""**Status:** ❌ **Analysis Failed**

Error: {str(e)}

Please check that the video and JSON files are valid."""
            self._update_unified_status(error_msg, '#f8d7da')
        
        finally:
            # Cleanup temp files
            for temp_file in ['temp_video_path', 'temp_json_path']:
                if temp_file in locals() and locals()[temp_file].exists():
                    try:
                        locals()[temp_file].unlink()
                    except:
                        pass
            self._reset_analysis_ui()
    
    def _process_with_progress(self, visualizer):
        """Process video with progress updates"""
        try:
            # Start the actual processing in a background thread
            import threading
            
            self.progress_thread_running = True
            
            def update_progress():
                # Simulate progress updates while processing
                for i in range(30, 90, 10):
                    if not self.is_analyzing or not self.progress_thread_running:  # Check if aborted or should stop
                        return
                    time.sleep(1)  # Update every second
                    self.analysis_progress.value = i
                    self._update_unified_status(f"**Status:** 🔄 Processing frames... ({i}%)\n\nAnalyzing video with tracking data...", '#fff3cd')
            
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
                self.analysis_progress.value = 95
                
        except Exception as e:
            raise e
    
    
    def _download_file(self, event):
        """Handle download button click"""
        if not self.current_output_path or not self.current_output_path.exists():
            return
        
        try:
            import base64
            
            with open(self.current_output_path, 'rb') as f:
                video_data = f.read()
            
            # Create a data URL for download
            b64_data = base64.b64encode(video_data).decode()
            
            # Create temporary download link and trigger it
            import panel as pn
            js_code = f"""
            const link = document.createElement('a');
            link.href = 'data:video/mp4;base64,{b64_data}';
            link.download = '{self.current_output_path.name}';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            """
            
            # Execute JavaScript to trigger download
            pn.io.push_notebook()
            pn.pane.HTML(f"<script>{js_code}</script>").servable()
            
        except Exception as e:
            self._update_unified_status(f"**Status:** ❌ Download Error\n\nCould not download file: {str(e)}", '#f8d7da')
    
    def _abort_analysis(self, event):
        """Abort the current analysis"""
        self.is_analyzing = False
        self._update_unified_status("**Status:** ❌ **Analysis Aborted by User**\n\nAnalysis process was cancelled.", '#f8d7da')
        self._reset_analysis_ui()
    
    def _reset_analysis_ui(self):
        """Reset UI to ready state"""
        self.analysis_progress.visible = False
        self.analysis_progress.value = 0
        
        # Check if we can enable start button
        has_video = self.video_input.value and len(self.video_input.value) > 0
        has_json = self.json_input.value and len(self.json_input.value) > 0
        self.start_analysis_button.disabled = not (has_video and has_json)
            
        self.abort_button.visible = False
        self.abort_button.disabled = True
        self.is_analyzing = False
        
        # Keep download button visible if analysis was completed
        if self.current_output_path and self.current_output_path.exists():
            self.download_button.visible = True
            self.download_button.disabled = False
        else:
            self.download_button.visible = False
            self.download_button.disabled = True
    
    def get_panel(self):
        """Return the main panel layout"""
        return pn.Column(
            pn.pane.Markdown("# 🧬 Ethological Analysis", margin=(0, 0, 20, 0)),
            
            # Video Tracking Analysis
            pn.pane.Markdown("## 📹 Video Tracking Analysis", margin=(0, 0, 10, 0)),
            
            pn.Row(
                # Left side - File inputs and options
                pn.Column(
                    self.video_input,
                    self.json_input,
                    pn.Spacer(height=15),
                    pn.pane.Markdown("**Options:**", margin=(0, 0, 5, 0)),
                    self.show_info_panel,
                    self.show_heatmap,
                    pn.Spacer(height=15),
                    pn.Row(
                        self.start_analysis_button,
                        self.abort_button,
                        self.download_button
                    ),
                    width=450
                ),
                
                pn.Spacer(width=20),
                
                # Right side - Status only
                pn.Column(
                    self.unified_status,
                    width=520
                )
            ),
            
            pn.Spacer(height=15),
            
            # Progress only
            self.analysis_progress,
            
            margin=(20, 20)
        )

def get_tab():
    """Create and return the Ethological Analysis tab"""
    ethological = EthologicalTab()
    return ethological.get_panel()