import panel as pn

from bokeh.models import BoxAnnotation, PolyAnnotation
from bokeh.plotting import figure
from bokeh.events import PanStart, Pan, PanEnd, Tap
from bokeh.io import curdoc

from PIL import Image
import numpy as np
import cv2 as cv

from threading import Thread, Timer
import zipfile
import math
import io
import os

# debug
import pprint

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
YOLO_RESOLUTION = (640, 640)
ROI_TYPES = ["Rectangle", "Polygon", "Circle"]
pn.extension()

class TrackingTab:
    def __init__(self):
        # select
        self.select_model_name  = None
        self.select_experiment_type  = None
        
        # list of models
        self.models_name= ["None"]
        
        # slider
        self.slider_confidence = None
        self.slider_iou = None
        
        # file input
        self.file_input = None
      
        # models dir
        self.models_dir = None
        
        # buttons
        self.button_start_tracking = None
        self.button_clear_roi = None
        
        # frames
        self.current_frame = np.ones(YOLO_RESOLUTION, dtype=np.uint8) * 240
        # cv.putText(self.current_frame, "No video available yet!", (240, 200), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1)
        self.frame_count = 0
        
        # store rois
        self.rois = []
        self.roi_count = 0
        self.select_roi = None
        
        # roi bounding box 
        self.bounding_box = BoxAnnotation(fill_alpha=0.3, fill_color='red')
        self.start_bounding_box = {'x': 0, 'y': 0}
        
        # roi polygon annotation 
        self.poly_annotation_points_draw = [] # store dot drawing
        self.poly_annotation_points_x = [] # store x-coordinates
        self.poly_annotation_points_y = [] # store y-coordinates
        
        # roi circle
        self.circle_annotation_points_draw = []
        self.circle_annotation_points = []
        self.circle_x_y_points = []
        
        # video
        self.video_loaded = False
        
        # progress bar
        self.progress_bar = pn.indicators.Progress(name='Progress', value=0, max=100, visible=False, width=620)
        
        # warning
        self.warning = pn.pane.Alert("## Alert\n", alert_type="danger", visible=False, width=620)
        
        # curdoc().add_periodic_callback(self._update_frames, 50)
        # self.thread_update = Thread(target=self._load_video)
        
        # tab settings 
        self._settings()
                
    def _settings(self):
        # selects
        self._yolo_models()
        self._experiment_type()
        self._select_roi()
        
        # sliders
        self._confidence_threshold()
        self._iou_threshold()
        
        # file input
        self._file_input()
    
        # models dir
        self._models_dir()        
        
        # buttons
        self._buttons()
    
        # frame display
        self.frame_pane = figure(width=640, height=640, tools="reset", x_range=(0, 640), y_range=(0,640))
        self.frame_pane.margin=0
        self.frame_pane.border_fill_color = "#808080"
        self.frame_pane.outline_line_color = None
        self.frame_pane.outline_line_width = 1

        # layout of figure tool bar
        self.frame_pane.xaxis.visible = False
        self.frame_pane.yaxis.visible = False
        self.frame_pane.grid.visible = False
        self.frame_pane.toolbar_location = None
        
        # initial image (gray)
        img = Image.fromarray(self.current_frame)
        
        # bokeh format
        img_array = np.array(img.transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA"))
        self.imview = img_array.view(np.uint32).reshape(img_array.shape[:2])

        self.current_frame = self.frame_pane.image_rgba(image=[self.imview], x=0, y=0, dw=640, dh=640)
        self.frame_view = None
                
        # connect functions
        self._connect_events()
      
    def _yolo_models(self):          
        self.select_model_name = pn.widgets.Select(
            name="Select a YOLO Model",
            options=self.models_name
            )
    
    def _models_dir(self):
        self.models_dir = pn.widgets.TextInput(
            name='📁 Models folder:', 
            value=MODELS_DIR, 
            width=620
        )
    
    def _experiment_type(self):
        self.experiment_type = ["EPM", "OF_Rectangle", "OF_Circular_1", "OF_Circular_2"]
        
        self.select_experiment_type = pn.widgets.Select(
            name="Experiment Type",
            options=self.experiment_type
            )
    
    def _select_roi(self):
        self.select_roi = pn.widgets.Select(
            name="ROI Type",
            options = ROI_TYPES
        )
        
    def _confidence_threshold(self):
        self.slider_confidence = pn.widgets.FloatSlider(
            name="Confidence Threshold",
            start=0.10,
            end=0.9,
            step=0.01,
            value=0.5
            )
            
    def _iou_threshold(self):
        self.slider_iou = pn.widgets.FloatSlider(
            name="IOU Threshold",
            start=0.10,
            end=0.9,
            step=0.01,
            value=0.5
            )
        
    def _file_input(self):
        self.file_input = pn.widgets.FileInput(
            name="Upload video",
            accept=".mp4, .mov, .avi, .mkv",
            multiple=False,
            height=40
            )
        
    def _buttons(self):        
        self.button_clear_roi = pn.widgets.Button(
            name="Clear ROI",
            button_type='danger',
            width=120,
            height=40,
            disabled=True
            )
        
        self.button_start_tracking = pn.widgets.Button(
            name="Start Tracking",
            button_type='success',
            width=120,
            height=40,
            disabled=True
            )
    
    def _thread_hide_warning(self):
        # hide warning
        self.hide_warning = Timer(3.0, self._hide_warning)
        self.hide_warning.start()
        
    def _connect_events(self):
        # load models
        self.models_dir.param.watch(lambda event: self._load_models(event.new), 'value')
        
        # load video
        self.file_input.param.watch(lambda event: self._load_video(event.new), 'value')
            
        # roi box
        self.frame_pane.on_event(PanStart, self._bb_pan_start)
        self.frame_pane.on_event(Pan, self._bb_pan)
        self.frame_pane.on_event(PanEnd, self._bb_pan_end)
        self.frame_pane.on_event(Tap, self._poly_annotation)
        self.frame_pane.on_event(Tap, self._circle_annotation)
      
        # clear roi  
        self.button_clear_roi.on_click(self._clear_roi)
    
    # ROI Functions---------------------------------------------------------------------------------------------------
    def _bb_pan_start(self, event):      
        if self.select_roi.value == "Rectangle":  
            if not self.video_loaded:
                self.warning.object = "## Alert\n Video is not loaded!"
                self.warning.visible = True
                self._thread_hide_warning()
                return
        
            self.roi_count += 1
            self.start_bounding_box['x'], self.start_bounding_box['y'] = event.x, event.y
            self.bounding_box.left = self.bounding_box.right = event.x
            self.bounding_box.bottom = self.bounding_box.top = event.y
   
    def _bb_pan(self, event):
        if self.video_loaded and self.select_roi.value == "Rectangle":
            self.bounding_box.left = min(self.start_bounding_box['x'], event.x)
            self.bounding_box.right = max(self.start_bounding_box['x'], event.x)
            self.bounding_box.top = min(self.start_bounding_box['y'], event.y)
            self.bounding_box.bottom = max(self.start_bounding_box['y'], event.y)
    
    def _bb_pan_end(self, event): 
        if self.video_loaded and self.select_roi.value == "Rectangle":       
            aux_box =  BoxAnnotation(fill_alpha=0.3, fill_color='red', top=self.bounding_box.top, bottom=self.bounding_box.bottom, right=self.bounding_box.right, left=self.bounding_box.left)   
            self.rois.append(aux_box)
            
            self.frame_pane.add_layout(aux_box)        
            
            self.button_clear_roi.disabled = False
            self.button_start_tracking.disabled = False
    
    def _poly_annotation(self, event):      
        if self.select_roi.value == "Polygon":
            if not self.video_loaded:
                self.warning.object = "## Alert\nOperation is invalid! Video is not loaded! "
                self.warning.visible = True
                self._thread_hide_warning()
                return      
            
            self.warning.visible = False
            self.button_clear_roi.disabled = False
            self.button_start_tracking.disabled = False
                
            # x and y coordinates (draw circle_dot)
            x = int(event.x)
            y = int(event.y)  
            
            # store points to draw a polygon in the future
            self.poly_annotation_points_x.append(event.x)
            self.poly_annotation_points_y.append(event.y)
            size_poly = len(self.poly_annotation_points_x)
            
            begin_x, begin_y = self.poly_annotation_points_x[0], self.poly_annotation_points_y[0]
            last_x, last_y = self.poly_annotation_points_x[size_poly-1], self.poly_annotation_points_y[size_poly-1]
            
            distance = math.sqrt((last_x-begin_x)**2 + (last_y-begin_y)**2)
                    
            # draw a polygon on the image
            if distance < 10 and size_poly>1:
                polygon = PolyAnnotation(
                fill_color="blue", fill_alpha=0.2,
                xs=self.poly_annotation_points_x,
                ys=self.poly_annotation_points_y,
                )
                
                self.frame_pane.add_layout(polygon)            
                
                # send the points and reset poly_annotations
                self.rois.append(polygon)
                self.roi_count += 1
                
                self.poly_annotation_points_x, self.poly_annotation_points_y = [], []
        
            else:
                dot = self.frame_pane.scatter(x, y, size=10, color="blue", marker="circle_dot", alpha=0.8)
                self.poly_annotation_points_draw.append(dot)
         
    def _circle_annotation(self, event):
        if self.select_roi.value == "Circle":
            # x and y coordinates
            x = event.x
            y = event.y
            
            dots_number = len(self.circle_annotation_points_draw)
            
            if dots_number < 2:
                dot = self.frame_pane.scatter(x, y, size=10, color="green", marker="circle_dot", alpha=0.8)
                self.circle_annotation_points.append(dot)
                self.circle_annotation_points_draw.append(dot)
                self.circle_x_y_points.append((x,y))
                
                dots_number = len(self.circle_annotation_points)

                if dots_number==2:
                    x0, y0 = self.circle_x_y_points[0]
                    x1, y1 = self.circle_x_y_points[1]
                    radius = (abs(x0-x1), abs(y0-y1))
                    
                    circle_draw = self.frame_pane.circle(x=x0, y=y0, radius=radius, radius_units='screen', color="green", alpha=0.3, hit_dilation=10.0)
                    self.rois.append(circle_draw)
                    
                    # clear dots storage
                    self.circle_annotation_points = []
                    self.circle_x_y_points = []
                    self.button_clear_roi.disabled = False               
               
    def _clear_roi(self, event):                 
        try:    
            # removes bounding box/polygon from figure                                
            if len(self.rois):
                if (self.select_roi.value == "Rectangle" or self.select_roi.value == "Polygon"):
                    for i in self.rois:
                        self.frame_pane.center.remove(i)
                
            # removes dots in a polygon from figure 
            if len(self.poly_annotation_points_draw):
                for i in self.poly_annotation_points_draw:
                    self.frame_pane.renderers.remove(i)
                        
                    self.poly_annotation_points_draw = []
                    
            # removes dots in a circle from figure 
            if len(self.circle_annotation_points_draw):
                for i in self.circle_annotation_points_draw:
                    self.frame_pane.renderers.remove(i)
                self.circle_annotation_points_draw = []
                
            self.rois = []
            self.roi_count = 0
            self.button_clear_roi.disabled = True
            self.button_start_tracking.disabled = True          
            
        except Exception as e:
            print(f"Error {e}")        
    #-----------------------------------------------------------------------------------------------------------------
    
    def _load_models(self, event):
        try:
            dir = os.listdir(event)
            self.models_name = []
        
            for file in dir:
                if file.endswith('.pt'):
                    self.models_name.append(file)
                    
            if self.models_dir:
                self.select_model_name.options = self.models_name
            else:
                print("nao contem .pt")
            
        except Exception as e:
            self.select_model_name.option = ["None"]
            print(f"Error {e}")
                    
    # Video Functions-------------------------------------------------------------------------------------------------
    def _load_video(self, event):
        mime_to_ext = {
            "video/mp4" : ".mp4",
            "video/avi" : ".avi"
        }
        
        if self.file_input.value is not None:
            video_format = self.file_input.mime_type 

            # tmp file to store video
            self.tmp_file = "./tmp_file" + mime_to_ext[video_format]
            
            try:
                with open(self.tmp_file, "wb") as tmp:
                    tmp.write(event)
                print("Temporary file created successfully")
                self.video_loaded = True
            except Exception as e:
                print(f"Error {e}")
                self.video_loaded = False
                return

            # only happens if the temp_file is created successfully
            if self.video_loaded:
                # self._collect_frames()
                self._calculate_background()
        
    def _calculate_background(self):
        try:
            cap = cv.VideoCapture(self.tmp_file)
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            
            if not cap.isOpened():
                self.warning.object = "## Alert\n Video couldn't be loaded!"
                self.warning.visible = True
                self._thread_hide_warning()
                return

            sample_ret, sample_frame = cap.read()
            
            if not sample_ret:
                self.warning.object ="## Alert\n Couldn't grab video info!"
                self.warning.visible=True
                self._thread_hide_warning()
                return
            
            # display progress bar
            self.progress_bar.visible = True
            self.progress_bar.value = 0
            
            # Reset video position
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            
            # Initialize frame accumulator
            # sample_frame = cv.resize(sample_frame, YOLO_RESOLUTION)
            frame_count = 0
            median_accumulator = np.zeros_like(sample_frame, dtype=np.float32)
        
            # Sample frames to calculate background (not processing every frame for efficiency)
            # Use frame sampling to process approximately 200 frames
            total_samples = min(200, total_frames)
            frame_step = max(1, total_frames // total_samples)

            height, width, _ = sample_frame.shape
            self.frame_pane.width = width
            self.frame_pane.height = height
            
            current_frame = 0
            
            while current_frame < total_frames:
                # Set position to the current frame
                cap.set(cv.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()

                # frame = cv.resize(frame, YOLO_RESOLUTION)
                
                if not ret:
                    break

                # Convert to float and accumulate
                frame_float = frame.astype(np.float32)
                median_accumulator += frame_float
                frame_count += 1
                
                # guarantee that warning is not visible when the progress bar is plotted
                self.warning.visible = False
                
                # update progress bar value
                self.progress_bar.value = int(current_frame/total_frames * 100)
                                
                # Move to next frame to sample
                current_frame += frame_step        
        
            self.progress_bar.visible = False
            cap.release()
            
            if frame_count == 0:
                self.warning.object = "No frames were processed"
                self.warning.visible = True
                self._thread_hide_warning()
                return

            # Calculate the average (approximating median for efficiency)
            background = (median_accumulator / frame_count).astype(np.uint8)

            # Save the background image
            cv.imwrite("background.png", background)
            
            # Convert to bokeh format
            frame_pil = Image.fromarray(background)
            frame_array = np.array(frame_pil.transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA"))
            self.frame_view = frame_array.view(np.uint32).reshape(frame_array.shape[:2])
            
            self.current_frame.data_source.data = {"image": [self.frame_view]}
                                
        except Exception as e:
            print(f"Background calculation error: {e}")       
    #-----------------------------------------------------------------------------------------------------------------
    
    def _hide_warning(self):
        self.warning.visible = False
        
    def get_panel(self):
        return pn.Column(pn.Row(pn.pane.Markdown("## Tracking\nTracking analysis tools will appear here.")), 
                         pn.Spacer(height=10),
                         pn.pane.Markdown("### ⚙️Settings") ,
                         self.models_dir,
                         pn.Row(pn.Column(self.select_model_name,  self.slider_confidence, self.file_input), pn.Column(self.select_experiment_type, self.slider_iou)),
                         pn.Spacer(height=20),
                         pn.pane.Markdown("### ROI Configuration"),
                         self.select_roi,
                         pn.Spacer(height=5),
                         pn.Row(self.button_start_tracking, pn.Spacer(width=10), self.button_clear_roi),
                         pn.Spacer(height=5),
                         self.warning,
                         self.progress_bar,
                         self.frame_pane,
                         margin=(10, 0))
                         
                             
def get_tab():
    track = TrackingTab()
    return track.get_panel()
