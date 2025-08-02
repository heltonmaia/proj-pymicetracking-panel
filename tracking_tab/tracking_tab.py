import panel as pn

from bokeh.models import BoxAnnotation
from bokeh.plotting import figure
from bokeh.events import PanStart, Pan, PanEnd
from bokeh.io import curdoc

from PIL import Image
import numpy as np
import cv2 as cv

from threading import Thread

class TrackingTab:
    def __init__(self):
        # select
        self.select_model_name  = None
        self.select_experiment_type  = None
        
        # slider
        self.slider_confidence = None
        self.slider_iou = None
        
        # file input
        self.file_input = None
        
        # buttons
        self.button_start_tracking = None
        self.button_clear_roi = None
        
        # frames
        self.current_frame = np.ones((480,640), dtype=np.uint8) * 240
        # cv.putText(self.current_frame, "No video available yet!", (240, 200), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1)
        self.frame_count = 0
        
        # store rois
        self.rois = []
        self.roi_count = 0
        
        # roi bounding box 
        self.bounding_box = BoxAnnotation(fill_alpha=0.3, fill_color='red')
        self.start_bounding_box = {'x': 0, 'y': 0}
        
        # video
        self.video_loaded = False
        
        # curdoc().add_periodic_callback(self._update_frames, 50)
        # self.thread_update = Thread(target=self._load_video)
        
        # tab settings 
        self._settings()
                
    def _settings(self):
        # selects
        self._yolo_models()
        self._experiment_type()
        
        # sliders
        self._confidence_threshold()
        self._iou_threshold()
        
        # file input
        self._file_input()
    
        # buttons
        self._buttons()
    
        # frame display
        self.frame_pane = figure(width=640, height=480, tools="reset", x_range=(0, 640), y_range=(0,480))
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

        self.current_frame = self.frame_pane.image_rgba(image=[self.imview], x=0, y=0, dw=640, dh=480)
        self.frame_view = None
        # connect functions
        self._connect_events()
    
    def _yolo_models(self):
        self.models_name= ["Default", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
            
        self.select_model_name = pn.widgets.Select(
            name="Select a YOLO Model",
            options=self.models_name
            )
    
    def _experiment_type(self):
        self.experiment_type = ["EPM", "OF_Rectangular", "OF_Circular_1", "OF_Circular_2"]
        
        self.select_experiment_type = pn.widgets.Select(
            name="Experiment Type",
            options=self.experiment_type
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
    
    def _connect_events(self):
        # load video
        self.file_input.param.watch(lambda event: self._load_video(event.new), 'value')
        
        # roi box
        self.frame_pane.on_event(PanStart, self._bb_pan_start)
        self.frame_pane.on_event(Pan, self._bb_pan)
        self.frame_pane.on_event(PanEnd, self._bb_pan_end)
        
        self.button_clear_roi.on_click(self._clear_roi)

        # clear roi
        
        # curdoc().add_periodic_callback(self._update_frames, 50)
    
    def _bb_pan_start(self, event):
        self.roi_count += 1
        print("Roi count:", self.roi_count)
        self.start_bounding_box['x'], self.start_bounding_box['y'] = event.x, event.y
        self.bounding_box.left = self.bounding_box.right = event.x
        self.bounding_box.bottom = self.bounding_box.top = event.y
   
    def _bb_pan(self, event):
        self.bounding_box.left = min(self.start_bounding_box['x'], event.x)
        self.bounding_box.right = max(self.start_bounding_box['x'], event.x)
        self.bounding_box.top = min(self.start_bounding_box['y'], event.y)
        self.bounding_box.bottom = max(self.start_bounding_box['y'], event.y)
        
   
    def _bb_pan_end(self, event):
        
        aux_box =  BoxAnnotation(fill_alpha=0.3, fill_color='red', top=self.bounding_box.top, bottom=self.bounding_box.bottom, right=self.bounding_box.right, left=self.bounding_box.left, name="123")   
        self.rois.append(aux_box)
        
        self.frame_pane.add_layout(aux_box)        
        # self.rois.append(aux_box.id, [aux_box.left, aux_box.right, aux_box.top, aux_box.bottom])
        
        self.button_clear_roi.disabled = False
        self.button_start_tracking.disabled = False
    
    def _clear_roi(self, event):  
        print(self.frame_pane.renderers)
        
        # for i in self.rois:
            # self.frame_pane._property_values["layout"].remove(i)

            # self.frame_pane.renderers.remove(i)
        
        # for r in self.frame_pane.renderers:
            # print(type(r), getattr(r, "glyph", None))      
        
        # self.rois = []     
        
        # boxes_id = self.frame_pane.select({'name': '123'})
        # self.frame_pane.renderers.remove()
                
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
                self._collect_frames()
                # self.thread_update.start()
              
    
    def _collect_frames(self):
        print("dento")
        cap = cv.VideoCapture(self.tmp_file)

        while cap.isOpened():
            print("doing")
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame)
            
            # bokeh format
            frame_array = np.array(frame_pil.transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA"))
            self.frame_view = frame_array.view(np.uint32).reshape(frame_array.shape[:2])
            
            self.current_frame.data_source.data["image"] = self.frame_view
            
        cap.release()
        self.video_loaded = False
        print("done")

        
    # def _update_frames(self):
    #     if self.video_loaded:
    #         print("atualiza")
    #         # self.current_frame.data_source.data["image"] = self.frame_view
        
        
    def get_panel(self):
        return pn.Column(pn.Row(pn.pane.Markdown("## Tracking\nTracking analysis tools will appear here.")), 
                         pn.Spacer(height=10),
                         pn.pane.Markdown("### ⚙️Settings") ,
                         pn.Row(pn.Column(self.select_model_name,  self.slider_confidence, self.file_input), pn.Column(self.select_experiment_type, self.slider_iou)),
                         pn.Spacer(height=10),
                         pn.pane.Markdown("### ROI Configuration") ,
                         pn.Row(self.button_start_tracking, pn.Spacer(width=40), self.button_clear_roi),
                         self.frame_pane,
                         margin=(10, 0))
                         
                             
def get_tab():
    track = TrackingTab()
    return track.get_panel()
