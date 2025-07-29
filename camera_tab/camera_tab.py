import panel as pn
import cv2
import numpy as np
import io
import base64
from PIL import Image
import os
from datetime import datetime

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments')

class CameraTab:
    def __init__(self):
        self.temp_video_file = None
        self.camera = None
        self.streaming = False
        self.recording = False
        self.video_writer = None
        self.frame_count = 0
        self.current_frame = None
        self.camera_callback = None

        # Frame display
        self.frame_pane = pn.pane.HTML(
            "<div style='width:640px; height:480px; border:2px solid #ccc; display:flex; align-items:center; justify-content:center; background:#f0f0f0; font-size:18px; color:#666;'>📹 Webcam off</div>",
            width=644,
            height=484
        )

        # Camera controls
        self.start_button = pn.widgets.Button(
            name='🎥 Start', 
            button_type='success',
            width=120,
            height=40
        )
        self.stop_button = pn.widgets.Button(
            name='⏹️ Stop', 
            button_type='danger',
            width=120,
            height=40
        )
        self.record_button = pn.widgets.Button(
            name='🔴 Record', 
            button_type='primary',
            width=120,
            height=40,
            disabled=True
        )
        self.stop_record_button = pn.widgets.Button(
            name='⏸️ Stop Rec.', 
            button_type='light',
            width=120,
            height=40,
            disabled=True
        )
        self.status_text = pn.pane.Markdown("**Status:** 🔴 Camera stopped")
        self.recording_text = pn.pane.Markdown("**Recording:** ⚪ Stopped")
        self.debug_text = pn.pane.Markdown("**Debug:** Waiting...")
        self.camera_info_text = pn.pane.Markdown("**Camera Info:** -")
        self.camera_select = pn.widgets.Select(name='📷 Camera:', options=self._get_available_cameras(), width=200)
        self.resolution_select = pn.widgets.Select(
            name='📐 Resolution:',
            options={
                '640x480': (640, 480),
                '800x600': (800, 600),
                '1280x720': (1280, 720),
                '1920x1080': (1920, 1080)
            },
            value=(640, 480), width=150
        )
        self.output_dir = pn.widgets.TextInput(
            name='📁 Output folder:', 
            value=EXPERIMENTS_DIR, 
            width=350
        )
        self._connect_events()

    def _get_available_cameras(self):
        index = 0
        arr = {}
        while index < 10:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                arr[f"Camera {index}"] = index
                cap.release()
            index += 1
        return arr if arr else {"No camera found": -1}

    def _connect_events(self):
        self.start_button.on_click(self.start_camera)
        self.stop_button.on_click(self.stop_camera)
        self.record_button.on_click(self.start_recording)
        self.stop_record_button.on_click(self.stop_recording)

    def start_camera(self, event=None):
        if self.streaming:
            return
        try:
            camera_index = self.camera_select.value
            if camera_index == -1:
                self.status_text.object = "**Status:** ❌ No camera selected."
                return
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                self.status_text.object = f"**Status:** ❌ Error opening camera {camera_index}."
                self.camera = None
                return
            # Apply selected resolution
            width_sel, height_sel = self.resolution_select.value
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width_sel)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height_sel)
            self.camera.set(cv2.CAP_PROP_FPS, 20)
            self.streaming = True

            # Update camera info
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_cap = self.camera.get(cv2.CAP_PROP_FPS)
            if fps_cap <= 0:
                fps_cap = 'N/A'
            self.camera_info_text.object = f"**Camera Info:** {width}x{height} @ {fps_cap} FPS"

            self.frame_count = 0
            self.status_text.object = "**Status:** 🟢 Camera active"
            self.record_button.disabled = False
            self.start_button.disabled = True
            self.stop_button.disabled = False
            self.camera_callback = pn.state.add_periodic_callback(
                self.update_camera_frame,
                period=50,  # About 20 FPS
                count=None
            )
        except Exception as e:
            self.status_text.object = f"**Status:** ❌ Error starting camera: {str(e)}"

    def stop_camera(self, event=None):
        if self.recording:
            self.stop_recording()
        self.streaming = False
        if self.camera_callback:
            self.camera_callback.stop()
            self.camera_callback = None
        if self.camera:
            self.camera.release()
            self.camera = None
        self.record_button.disabled = True
        self.stop_record_button.disabled = True
        self.start_button.disabled = False
        self.stop_button.disabled = True
        self.status_text.object = "**Status:** 🔴 Camera stopped"
        self.recording_text.object = "**Recording:** ⚪ Stopped"
        self.debug_text.object = "**Debug:** Camera off"

    def start_recording(self, event=None):
        if not self.streaming or self.recording:
            return
        self.recording = True
        
        # Ensure output folder exists
        output_folder = self.output_dir.value
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                self.debug_text.object = f"**Debug:** Folder created: {output_folder}"
            except Exception as e:
                self.debug_text.object = f"**Debug:** ❌ Error creating folder: {e}"
                self.recording = False
                return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Scale video to current camera resolution
        width_cap = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_cap = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.temp_video_file = os.path.join(output_folder, f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20.0
        frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = cv2.VideoWriter(self.temp_video_file, fourcc, fps, (width_cap, height_cap))
        self.recording_text.object = "**Recording:** 🔴 Recording..."
        self.record_button.disabled = True
        self.stop_record_button.disabled = False

    def stop_recording(self, event=None):
        if not self.recording:
            return
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.temp_video_file and os.path.exists(self.temp_video_file):
            final_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_folder = self.output_dir.value
            final_filepath = os.path.join(output_folder, final_filename)
            try:
                os.rename(self.temp_video_file, final_filepath)
                self.debug_text.object = f"**Debug:** ✅ Video saved to {final_filepath}"
            except Exception as e:
                self.debug_text.object = f"**Debug:** ❌ Error saving video: {e}"
        self.recording_text.object = "**Recording:** ⚪ Stopped"
        self.record_button.disabled = False
        self.stop_record_button.disabled = True

    def update_camera_frame(self):
        if not self.camera or not self.streaming:
            return
        try:
            ret, frame = self.camera.read()
            if ret:
                self.frame_count += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (640, 480))
                self.current_frame = cv2.resize(frame, (640, 480))
                
                # Gravar frame se estiver gravando
                if self.recording and self.video_writer:
                    self.video_writer.write(self.current_frame)
                
                img_data = self.numpy_to_base64(frame_rgb)
                if img_data:
                    html_content = f"""
                    <div style='width:640px; height:480px; border:3px solid #2980B9; border-radius:8px; overflow:hidden;'>
                        <img src="{img_data}" style="width:100%; height:100%; object-fit:cover;" alt="Live webcam"/>
                    </div>
                    """
                    self.frame_pane.object = html_content
                    self.debug_text.object = f"**Debug:** Frame {self.frame_count}"
        except Exception as e:
            self.debug_text.object = f"**Debug:** Camera error: {str(e)}"

    def numpy_to_base64(self, frame):
        try:
            pil_image = Image.fromarray(frame)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            img_data = buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            print(f"Error in conversion: {e}")
            return None

    def get_panel(self):
        return pn.Column(
            pn.Row(self.camera_select, self.resolution_select),
            pn.Spacer(height=10),
            pn.Row(self.output_dir),
            pn.Spacer(height=15),
            pn.Row(
                self.start_button,
                self.stop_button,
                pn.Spacer(width=40),
                self.record_button,
                self.stop_record_button,
                align='center',
            ),
            pn.Spacer(height=10),
            self.frame_pane,
            self.status_text,
            self.camera_info_text,
            self.recording_text,
            self.debug_text,
            margin=(10, 0)
        )

def get_tab():
    cam = CameraTab()
    return cam.get_panel()
