import panel as pn
import os
import json
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import ndimage

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
            width=550,
            height=250
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
        
        # Movement Heatmap Analysis components
        self.heatmap_json_input = pn.widgets.FileInput(
            accept='.json',
            name='📊 Select Tracking JSON:',
            width=400,
            multiple=False
        )
        
        self.export_format = pn.widgets.RadioBoxGroup(
            name='Export Format:',
            value='png',
            options=['png', 'eps'],
            inline=True,
            width=200
        )
        
        # Heatmap configuration
        self.heatmap_bins = pn.widgets.IntSlider(
            name='Heatmap Resolution (bins):',
            start=20,
            end=100,
            step=10,
            value=50,
            width=300
        )
        
        self.heatmap_colormap = pn.widgets.Select(
            name='Heatmap Colormap:',
            value='hot',
            options=['hot', 'viridis', 'plasma', 'jet', 'rainbow', 'coolwarm'],
            width=200
        )
        
        self.heatmap_alpha = pn.widgets.FloatSlider(
            name='Heatmap Transparency:',
            start=0.5,
            end=1.0,
            step=0.1,
            value=0.8,
            width=250
        )
        
        # Movement analysis configuration
        self.movement_threshold_percentile = pn.widgets.IntSlider(
            name='Movement Threshold (percentile):',
            start=50,
            end=95,
            step=5,
            value=75,
            width=300
        )
        
        self.velocity_bins = pn.widgets.IntSlider(
            name='Velocity Histogram Bins:',
            start=15,
            end=50,
            step=5,
            value=30,
            width=300
        )
        
        # Analysis type selection
        self.analysis_type = pn.widgets.RadioBoxGroup(
            name='Analysis Type:',
            value='complete',
            options=[('complete', 'Complete Panel'), ('individual', 'Individual Plots')],
            inline=False,
            width=350
        )
        
        self.generate_analysis_button = pn.widgets.Button(
            name='📊 Generate Analysis',
            button_type='primary',
            width=200,
            height=40,
            disabled=True
        )
        
        # Download button for generated image
        self.download_heatmap_button = pn.widgets.Button(
            name='💾 Download Image',
            button_type='success',
            width=150,
            height=40,
            disabled=True,
            visible=False
        )
        
        # Store current analysis output path
        self.current_analysis_path = None
        
        self.heatmap_status = pn.pane.Markdown(
            "**Status:** Ready\n\nSelect a tracking JSON file to start analysis.",
            styles={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '5px'},
            width=550,
            height=250
        )
        
        # Connect events
        self.video_input.param.watch(self._on_file_selected, 'value')
        self.video_input.param.watch(self._on_file_selected, 'filename')
        self.json_input.param.watch(self._on_file_selected, 'value')
        self.json_input.param.watch(self._on_file_selected, 'filename')
        self.heatmap_json_input.param.watch(self._on_heatmap_json_selected, 'value')
        self.heatmap_json_input.param.watch(self._on_heatmap_json_selected, 'filename')
        self.start_analysis_button.on_click(self._start_analysis)
        self.abort_button.on_click(self._abort_analysis)
        self.download_button.on_click(self._download_file)
        self.generate_analysis_button.on_click(self._generate_complete_analysis)
        self.download_heatmap_button.on_click(self._download_analysis_image)
    
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
    
    def _on_heatmap_json_selected(self, event):
        """Handle heatmap JSON file selection"""
        self._update_heatmap_status()
    
    def _update_heatmap_status(self):
        """Update heatmap analysis status based on selected JSON"""
        has_json = self.heatmap_json_input.value and len(self.heatmap_json_input.value) > 0
        
        if not has_json:
            self._update_heatmap_status_panel("**Status:** Ready\n\nSelect a tracking JSON file to start analysis.", '#f8f9fa')
            self.generate_analysis_button.disabled = True
            return
        
        # Validate JSON file
        try:
            # Create temporary JSON file to validate
            temp_json_path = self.project_root / 'temp_heatmap_json_for_validation.json'
            with open(temp_json_path, 'wb') as f:
                f.write(self.heatmap_json_input.value)
            
            if self._validate_json_file(temp_json_path):
                json_filename = getattr(self.heatmap_json_input, 'filename', 'data.json')
                json_size_kb = len(self.heatmap_json_input.value) / 1024
                
                status_msg = f"**Status:** ✅ Ready to analyze\n\n**JSON:** {json_filename} ({json_size_kb:.1f} KB)\n\nConfigure parameters and click Generate Complete Analysis."
                self._update_heatmap_status_panel(status_msg, '#d1ecf1')
                
                self.generate_analysis_button.disabled = False
            else:
                self._update_heatmap_status_panel("**Status:** ❌ Invalid JSON format\n\nThe JSON file doesn't contain valid tracking data.", '#f8d7da')
                self.generate_analysis_button.disabled = True
            
            # Cleanup temp file
            if temp_json_path.exists():
                temp_json_path.unlink()
                
        except Exception as e:
            self._update_heatmap_status_panel(f"**Status:** ❌ Error validating JSON\n\n{str(e)}", '#f8d7da')
            self.generate_analysis_button.disabled = True
    
    def _update_heatmap_status_panel(self, message: str, background_color: str):
        """Update heatmap status panel"""
        self.heatmap_status.object = message
        self.heatmap_status.styles = {'background': background_color, 'padding': '15px', 'border-radius': '5px'}
    
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
    
    def _generate_complete_analysis(self, event):
        """Generate complete movement analysis with heatmap and statistics"""
        if not self.heatmap_json_input.value:
            return
            
        try:
            analysis_mode = self.analysis_type.value
            if analysis_mode == 'complete':
                self._update_heatmap_status_panel("**Status:** 🔄 Generating complete analysis panel...\n\nProcessing tracking data...", '#fff3cd')
            else:
                self._update_heatmap_status_panel("**Status:** 🔄 Generating individual plots...\n\nProcessing tracking data...", '#fff3cd')
            
            # Load tracking data
            temp_json_path = self.temp_dir / 'temp_complete_analysis.json'
            with open(temp_json_path, 'wb') as f:
                f.write(self.heatmap_json_input.value)
            
            with open(temp_json_path, 'r') as f:
                data = json.load(f)
            
            tracking_data = data.get("tracking_data", [])
            if not tracking_data:
                self._update_heatmap_status_panel("**Status:** ❌ No tracking data found", '#f8d7da')
                return
            
            # Extract positions and frame numbers
            frames = []
            x_positions = []
            y_positions = []
            
            for frame_data in tracking_data:
                if all(key in frame_data for key in ['frame_number', 'centroid_x', 'centroid_y']):
                    frames.append(frame_data['frame_number'])
                    x_positions.append(frame_data['centroid_x'])
                    y_positions.append(frame_data['centroid_y'])
            
            if not frames:
                self._update_heatmap_status_panel("**Status:** ❌ No position data found", '#f8d7da')
                return
            
            # Calculate movement metrics
            x_positions = np.array(x_positions)
            y_positions = np.array(y_positions)
            
            # Calculate distances from center of mass
            center_x = np.mean(x_positions)
            center_y = np.mean(y_positions)
            distances_from_center = np.sqrt((x_positions - center_x)**2 + (y_positions - center_y)**2)
            
            # Calculate velocity (frame-to-frame movement)
            velocities = []
            for i in range(1, len(x_positions)):
                dx = x_positions[i] - x_positions[i-1]
                dy = y_positions[i] - y_positions[i-1]
                velocity = np.sqrt(dx**2 + dy**2)
                velocities.append(velocity)
            
            json_filename = getattr(self.heatmap_json_input, 'filename', 'data.json')
            base_name = Path(json_filename).stem
            format_ext = self.export_format.value
            
            if analysis_mode == 'individual':
                # Generate individual plots and save them separately
                self._generate_individual_plots(x_positions, y_positions, frames, velocities, 
                                               distances_from_center, center_x, center_y, 
                                               base_name, format_ext)
            else:
                # Generate complete panel
                self._generate_complete_panel(x_positions, y_positions, frames, velocities, 
                                            distances_from_center, center_x, center_y, 
                                            base_name, format_ext)
            
            # Cleanup temp file
            if temp_json_path.exists():
                temp_json_path.unlink()
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self._update_heatmap_status_panel(f"**Status:** ❌ Error generating analysis\n\n{str(e)}\n\nDetails: {error_details[:200]}", '#f8d7da')
    
    def _generate_complete_panel(self, x_positions, y_positions, frames, velocities, 
                               distances_from_center, center_x, center_y, base_name, format_ext):
        """Generate complete analysis panel with all plots"""
        # Create comprehensive analysis figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1.5, 1.5], width_ratios=[2, 1, 1])
        
        # Calculate additional statistics
        total_distance = np.sum(velocities) if velocities else 0
        max_distance_from_center = np.max(distances_from_center)
        movement_threshold = np.percentile(velocities, self.movement_threshold_percentile.value) if velocities else 0
        moving_frames = np.array(velocities) > movement_threshold if velocities else []
        stationary_ratio = 1 - (np.sum(moving_frames) / len(moving_frames)) if len(moving_frames) > 0 else 1
        
        # Plot 1: High-resolution Movement Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        bins = self.heatmap_bins.value
        heatmap, xedges, yedges = np.histogram2d(x_positions, y_positions, bins=bins, density=True)
        heatmap_smooth = ndimage.gaussian_filter(heatmap, sigma=1.0)
        
        cmap = self.heatmap_colormap.value
        alpha = self.heatmap_alpha.value
        
        im = ax1.imshow(heatmap_smooth.T, origin='lower', 
                       extent=[min(x_positions), max(x_positions), min(y_positions), max(y_positions)], 
                       cmap=cmap, aspect='equal', alpha=alpha, interpolation='bilinear')
        
        ax1.plot(x_positions, y_positions, 'k-', alpha=0.3, linewidth=0.5, label='Trajectory')
        ax1.scatter(center_x, center_y, c='red', s=100, marker='x', linewidth=3, label='Center of Mass')
        
        plt.colorbar(im, ax=ax1, label='Movement Density', shrink=0.6)
        ax1.set_title('Animal Movement Heatmap', fontsize=16, fontweight='bold')
        ax1.set_xlabel('X Position (pixels)', fontsize=12)
        ax1.set_ylabel('Y Position (pixels)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Statistics Summary
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        stats_text = f"""Movement Analysis Summary

Total Frames: {len(frames)}
Analysis Duration: {len(frames)} frames

Spatial Statistics:
• Center of Mass: ({center_x:.1f}, {center_y:.1f})
• Mean distance from center: {np.mean(distances_from_center):.1f}px
• Max distance from center: {max_distance_from_center:.1f}px

Movement Statistics:
• Total distance traveled: {total_distance:.1f}px
• Mean velocity: {np.mean(velocities) if velocities else 0:.1f}px/frame
• Max velocity: {np.max(velocities) if velocities else 0:.1f}px/frame
• Movement threshold: {movement_threshold:.1f}px/frame
• Time stationary: {stationary_ratio*100:.1f}%
• Time moving: {(1-stationary_ratio)*100:.1f}%

Configuration:
• Heatmap bins: {bins}
• Colormap: {cmap}
• Transparency: {alpha}
• Movement threshold: {self.movement_threshold_percentile.value}th percentile"""
        
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Plot 3: Distance from center
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(frames, distances_from_center, 'b-', linewidth=1, alpha=0.7)
        ax3.axhline(y=np.mean(distances_from_center), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(distances_from_center):.1f}px')
        ax3.set_title('Distance from Center of Mass', fontweight='bold')
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Distance (pixels)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Movement velocity
        ax4 = fig.add_subplot(gs[1, 1])
        if velocities:
            velocity_frames = frames[1:]
            ax4.plot(velocity_frames, velocities, 'g-', linewidth=1, alpha=0.7)
            ax4.axhline(y=np.mean(velocities), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(velocities):.1f}px/frame')
            ax4.axhline(y=movement_threshold, color='orange', linestyle=':', 
                       label=f'Movement threshold')
            ax4.set_title('Movement Velocity', fontweight='bold')
            ax4.set_xlabel('Frame Number')
            ax4.set_ylabel('Velocity (pixels/frame)')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        # Plot 5: Velocity distribution
        ax5 = fig.add_subplot(gs[1, 2])
        if velocities:
            bins_hist = self.velocity_bins.value
            ax5.hist(velocities, bins=bins_hist, alpha=0.7, color='purple', edgecolor='black')
            ax5.axvline(x=np.mean(velocities), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(velocities):.1f}px/frame')
            ax5.axvline(x=movement_threshold, color='orange', linestyle=':', 
                       label=f'Movement threshold')
            ax5.set_title('Velocity Distribution', fontweight='bold')
            ax5.set_xlabel('Velocity (pixels/frame)')
            ax5.set_ylabel('Frequency')
            ax5.legend()
        
        # Plot 6: Activity classification
        ax6 = fig.add_subplot(gs[2, 0])
        if velocities:
            labels = ['Moving', 'Stationary']
            sizes = [1 - stationary_ratio, stationary_ratio]
            colors = ['#ff9999', '#66b3ff']
            ax6.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax6.set_title('Activity Classification', fontweight='bold')
        
        # Plot 7: Movement directions
        ax7 = fig.add_subplot(gs[2, 1], projection='polar')
        if len(x_positions) > 1:
            angles = []
            for i in range(1, len(x_positions)):
                dx = x_positions[i] - x_positions[i-1]
                dy = y_positions[i] - y_positions[i-1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                angles.append(angle)
            
            theta = np.array(angles) * np.pi / 180
            ax7.hist(theta, bins=16, alpha=0.7, color='green')
            ax7.set_title('Movement Directions', fontweight='bold', pad=20)
            ax7.set_theta_zero_location('E')
            ax7.set_theta_direction(1)
        
        # Plot 8: Cumulative distance
        ax8 = fig.add_subplot(gs[2, 2])
        if velocities:
            cumulative_distance = np.cumsum(velocities)
            ax8.plot(frames[1:], cumulative_distance, 'orange', linewidth=2)
            ax8.set_title('Cumulative Distance', fontweight='bold')
            ax8.set_xlabel('Frame Number')
            ax8.set_ylabel('Cumulative Distance (pixels)')
            ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save complete panel
        output_path = self.temp_dir / f"{base_name}_complete_analysis.{format_ext}"
        dpi = 300
        if format_ext == 'eps':
            plt.savefig(output_path, format='eps', dpi=dpi, bbox_inches='tight')
        else:
            plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
        
        plt.close()
        
        # Store path and enable download
        self.current_analysis_path = output_path
        self.download_heatmap_button.disabled = False
        self.download_heatmap_button.visible = True
        
        # Success message
        file_size_kb = output_path.stat().st_size / 1024
        success_msg = f"""**Status:** ✅ Complete Analysis Generated

**File:** {output_path.name} ({file_size_kb:.1f} KB)
**Location:** ethological_tab/temp/

**Analysis Includes:**
• High-resolution movement heatmap
• Distance from center of mass analysis
• Movement velocity analysis
• Activity classification
• Movement direction analysis
• Cumulative distance tracking
• Comprehensive statistics summary

**Key Results:**
• Total distance: {total_distance:.1f}px
• Stationary time: {stationary_ratio*100:.1f}%
• Mean velocity: {np.mean(velocities) if velocities else 0:.1f}px/frame"""
        
        self._update_heatmap_status_panel(success_msg, '#d4edda')
    
    def _generate_individual_plots(self, x_positions, y_positions, frames, velocities, 
                                 distances_from_center, center_x, center_y, base_name, format_ext):
        """Generate individual plots and save them separately"""
        dpi = 300
        plot_count = 0
        
        # Calculate statistics
        total_distance = np.sum(velocities) if velocities else 0
        movement_threshold = np.percentile(velocities, self.movement_threshold_percentile.value) if velocities else 0
        moving_frames = np.array(velocities) > movement_threshold if velocities else []
        stationary_ratio = 1 - (np.sum(moving_frames) / len(moving_frames)) if len(moving_frames) > 0 else 1
        
        # 1. Heatmap
        plt.figure(figsize=(12, 8))
        bins = self.heatmap_bins.value
        heatmap, xedges, yedges = np.histogram2d(x_positions, y_positions, bins=bins, density=True)
        heatmap_smooth = ndimage.gaussian_filter(heatmap, sigma=1.0)
        
        cmap = self.heatmap_colormap.value
        alpha = self.heatmap_alpha.value
        
        im = plt.imshow(heatmap_smooth.T, origin='lower', 
                       extent=[min(x_positions), max(x_positions), min(y_positions), max(y_positions)], 
                       cmap=cmap, aspect='equal', alpha=alpha, interpolation='bilinear')
        
        plt.plot(x_positions, y_positions, 'k-', alpha=0.3, linewidth=0.5, label='Trajectory')
        plt.scatter(center_x, center_y, c='red', s=100, marker='x', linewidth=3, label='Center of Mass')
        
        plt.colorbar(im, label='Movement Density')
        plt.title('Animal Movement Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('X Position (pixels)', fontsize=12)
        plt.ylabel('Y Position (pixels)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        output_path = self.temp_dir / f"{base_name}_01_heatmap.{format_ext}"
        if format_ext == 'eps':
            plt.savefig(output_path, format='eps', dpi=dpi, bbox_inches='tight')
        else:
            plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
        plt.close()
        plot_count += 1
        
        # 2. Distance from center
        plt.figure(figsize=(10, 6))
        plt.plot(frames, distances_from_center, 'b-', linewidth=1, alpha=0.7)
        plt.axhline(y=np.mean(distances_from_center), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(distances_from_center):.1f}px')
        plt.title('Distance from Center of Mass', fontsize=16, fontweight='bold')
        plt.xlabel('Frame Number')
        plt.ylabel('Distance (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        output_path = self.temp_dir / f"{base_name}_02_distance_from_center.{format_ext}"
        if format_ext == 'eps':
            plt.savefig(output_path, format='eps', dpi=dpi, bbox_inches='tight')
        else:
            plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
        plt.close()
        plot_count += 1
        
        # Continue with other individual plots...
        if velocities:
            # 3. Movement velocity
            plt.figure(figsize=(10, 6))
            velocity_frames = frames[1:]
            plt.plot(velocity_frames, velocities, 'g-', linewidth=1, alpha=0.7)
            plt.axhline(y=np.mean(velocities), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(velocities):.1f}px/frame')
            plt.axhline(y=movement_threshold, color='orange', linestyle=':', 
                       label=f'Movement threshold')
            plt.title('Movement Velocity', fontsize=16, fontweight='bold')
            plt.xlabel('Frame Number')
            plt.ylabel('Velocity (pixels/frame)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            output_path = self.temp_dir / f"{base_name}_03_velocity.{format_ext}"
            if format_ext == 'eps':
                plt.savefig(output_path, format='eps', dpi=dpi, bbox_inches='tight')
            else:
                plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
            plt.close()
            plot_count += 1
            
            # 4. Velocity distribution
            plt.figure(figsize=(8, 6))
            bins_hist = self.velocity_bins.value
            plt.hist(velocities, bins=bins_hist, alpha=0.7, color='purple', edgecolor='black')
            plt.axvline(x=np.mean(velocities), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(velocities):.1f}px/frame')
            plt.axvline(x=movement_threshold, color='orange', linestyle=':', 
                       label=f'Movement threshold')
            plt.title('Velocity Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Velocity (pixels/frame)')
            plt.ylabel('Frequency')
            plt.legend()
            
            output_path = self.temp_dir / f"{base_name}_04_velocity_distribution.{format_ext}"
            if format_ext == 'eps':
                plt.savefig(output_path, format='eps', dpi=dpi, bbox_inches='tight')
            else:
                plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
            plt.close()
            plot_count += 1
            
            # 5. Activity classification
            plt.figure(figsize=(8, 8))
            labels = ['Moving', 'Stationary']
            sizes = [1 - stationary_ratio, stationary_ratio]
            colors = ['#ff9999', '#66b3ff']
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Activity Classification', fontsize=16, fontweight='bold')
            
            output_path = self.temp_dir / f"{base_name}_05_activity_classification.{format_ext}"
            if format_ext == 'eps':
                plt.savefig(output_path, format='eps', dpi=dpi, bbox_inches='tight')
            else:
                plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
            plt.close()
            plot_count += 1
            
            # 6. Cumulative distance
            plt.figure(figsize=(10, 6))
            cumulative_distance = np.cumsum(velocities)
            plt.plot(frames[1:], cumulative_distance, 'orange', linewidth=2)
            plt.title('Cumulative Distance', fontsize=16, fontweight='bold')
            plt.xlabel('Frame Number')
            plt.ylabel('Cumulative Distance (pixels)')
            plt.grid(True, alpha=0.3)
            
            output_path = self.temp_dir / f"{base_name}_06_cumulative_distance.{format_ext}"
            if format_ext == 'eps':
                plt.savefig(output_path, format='eps', dpi=dpi, bbox_inches='tight')
            else:
                plt.savefig(output_path, format='png', dpi=dpi, bbox_inches='tight')
            plt.close()
            plot_count += 1
        
        # Store the last path for download (or create a zip?)
        self.current_analysis_path = self.temp_dir / f"{base_name}_01_heatmap.{format_ext}"
        self.download_heatmap_button.disabled = False
        self.download_heatmap_button.visible = True
        
        # Success message
        success_msg = f"""**Status:** ✅ Individual Plots Generated

**Files Generated:** {plot_count} individual plots
**Location:** ethological_tab/temp/
**Format:** {format_ext.upper()}

**Plots Include:**
• {base_name}_01_heatmap.{format_ext}
• {base_name}_02_distance_from_center.{format_ext}
• {base_name}_03_velocity.{format_ext}
• {base_name}_04_velocity_distribution.{format_ext}
• {base_name}_05_activity_classification.{format_ext}
• {base_name}_06_cumulative_distance.{format_ext}

**Key Results:**
• Total distance: {total_distance:.1f}px
• Stationary time: {stationary_ratio*100:.1f}%
• Mean velocity: {np.mean(velocities) if velocities else 0:.1f}px/frame"""
        
        self._update_heatmap_status_panel(success_msg, '#d4edda')
    
    def _download_analysis_image(self, event):
        """Handle download button click for analysis images"""
        if not self.current_analysis_path or not self.current_analysis_path.exists():
            return
        
        try:
            import base64
            
            with open(self.current_analysis_path, 'rb') as f:
                image_data = f.read()
            
            # Create a data URL for download
            b64_data = base64.b64encode(image_data).decode()
            
            # Determine MIME type
            if self.current_analysis_path.suffix.lower() == '.eps':
                mime_type = 'application/postscript'
            else:
                mime_type = 'image/png'
            
            # Create download link
            js_code = f"""
            const link = document.createElement('a');
            link.href = 'data:{mime_type};base64,{b64_data}';
            link.download = '{self.current_analysis_path.name}';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            """
            
            # Execute JavaScript to trigger download
            import panel as pn
            pn.io.push_notebook()
            pn.pane.HTML(f"<script>{js_code}</script>").servable()
            
        except Exception as e:
            self._update_heatmap_status_panel(f"**Status:** ❌ Download Error\n\nCould not download file: {str(e)}", '#f8d7da')
    
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
                    # File Selection with background
                    pn.Column(
                        pn.pane.Markdown("**File Selection:**", margin=(5, 5, 5, 5)),
                        self.video_input,
                        self.json_input,
                        styles={'background': '#fff3e0', 'padding': '10px', 'border-radius': '8px', 'margin': '5px'}
                    ),
                    
                    pn.Spacer(height=15),
                    
                    # Analysis Options with background
                    pn.Column(
                        pn.pane.Markdown("**Analysis Options:**", margin=(5, 5, 5, 5)),
                        self.show_info_panel,
                        self.show_heatmap,
                        styles={'background': '#fce4ec', 'padding': '10px', 'border-radius': '8px', 'margin': '5px'}
                    ),
                    
                    pn.Spacer(height=15),
                    
                    # Analysis Controls with background
                    pn.Column(
                        pn.pane.Markdown("**Analysis Controls:**", margin=(5, 5, 5, 5)),
                        pn.Row(
                            self.start_analysis_button,
                            self.abort_button,
                            self.download_button
                        ),
                        styles={'background': '#e0f2f1', 'padding': '10px', 'border-radius': '8px', 'margin': '5px'}
                    ),
                    
                    width=480
                ),
                
                pn.Spacer(width=20),
                
                # Right side - Status only
                pn.Column(
                    self.unified_status,
                    width=570
                )
            ),
            
            pn.Spacer(height=15),
            
            # Progress only
            self.analysis_progress,
            
            pn.Spacer(height=30),
            pn.pane.Markdown("---"),
            pn.Spacer(height=20),
            
            # Movement Heatmap Analysis
            pn.pane.Markdown("## 🔥 Movement Heatmap Analysis", margin=(0, 0, 10, 0)),
            
            pn.Row(
                # Left side - File inputs and configuration
                pn.Column(
                    self.heatmap_json_input,
                    pn.Spacer(height=15),
                    
                    # Heatmap Configuration with background
                    pn.Column(
                        pn.pane.Markdown("**Heatmap Configuration:**", margin=(5, 5, 5, 5)),
                        self.heatmap_bins,
                        pn.Row(
                            self.heatmap_colormap,
                            pn.Spacer(width=10),
                            self.heatmap_alpha
                        ),
                        styles={'background': '#e3f2fd', 'padding': '10px', 'border-radius': '8px', 'margin': '5px'}
                    ),
                    
                    pn.Spacer(height=15),
                    
                    # Movement Analysis Configuration with background
                    pn.Column(
                        pn.pane.Markdown("**Movement Analysis Configuration:**", margin=(5, 5, 5, 5)),
                        self.movement_threshold_percentile,
                        self.velocity_bins,
                        styles={'background': '#f3e5f5', 'padding': '10px', 'border-radius': '8px', 'margin': '5px'}
                    ),
                    
                    pn.Spacer(height=15),
                    
                    # Export and Analysis Options with background
                    pn.Column(
                        pn.pane.Markdown("**Export & Analysis Options:**", margin=(5, 5, 10, 5)),
                        
                        pn.pane.Markdown("**Export Format:**", margin=(0, 0, 5, 0)),
                        self.export_format,
                        
                        pn.Spacer(height=15),
                        
                        pn.pane.Markdown("**Analysis Mode:**", margin=(0, 0, 5, 0)),
                        self.analysis_type,
                        
                        pn.Spacer(height=20),
                        
                        pn.pane.Markdown("**Actions:**", margin=(0, 0, 10, 0)),
                        pn.Row(
                            self.generate_analysis_button,
                            pn.Spacer(width=10),
                            self.download_heatmap_button,
                        ),
                        
                        styles={'background': '#e8f5e8', 'padding': '15px', 'border-radius': '8px', 'margin': '5px'},
                        width=460
                    ),
                    
                    width=500
                ),
                
                pn.Spacer(width=20),
                
                # Right side - Status
                pn.Column(
                    self.heatmap_status,
                    width=570
                )
            ),
            
            margin=(20, 20)
        )

def get_tab():
    """Create and return the Ethological Analysis tab"""
    ethological = EthologicalTab()
    return ethological.get_panel()