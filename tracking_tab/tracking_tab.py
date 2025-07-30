import panel as pn

class TrackingTab:
    def __init__(self):
        self.select_model_name  = None
        self.select_experiment_type  = None
        self.slider_confidence = None
        self.slider_iou = None
        self.file_input = None
        self.button_load_settings = None
        self.button_start_tracking = None
        self.button_clear_roi = None
        self.button_done_roi = None
        self.frame_count = 0
        self.current_frame = None  
              
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
        self.frame_pane = pn.pane.HTML(
            "<div style='width:640px; height:480px; border:2px solid #ccc; display:flex; align-items:center; justify-content:center; background:#f0f0f0; font-size:18px; color:#666;'>🎥 No video yet</div>",
            width=644,
            height=484
        )
        
    
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
        self.button_done_roi = pn.widgets.Button(
            name="ROI Done",
            button_type='primary',
            width=120,
            height=40
            )
        
        self.button_clear_roi = pn.widgets.Button(
            name="Clear ROI",
            button_type='danger',
            width=120,
            height=40
            )
        
        self.button_start_tracking = pn.widgets.Button(
            name="Start Tracking",
            button_type='success',
            width=120,
            height=40
            )
        
        self.button_load_settings = pn.widgets.Button(
            name="Load Settings",
            width=300,
            height=40
            )
    
    def _connect_events(self):
        pass
    
    def get_panel(self):
        return pn.Column(pn.Row(pn.pane.Markdown("## Tracking\nTracking analysis tools will appear here.")), 
                         pn.Spacer(height=10),
                         pn.pane.Markdown("### ⚙️Settings") ,
                         pn.Row(pn.Column(self.select_model_name,  self.slider_confidence, self.file_input), pn.Column(self.select_experiment_type, self.slider_iou, self.button_load_settings)),
                         pn.Spacer(height=10),
                         pn.pane.Markdown("### ROI Configuration") ,
                         pn.Row(self.button_start_tracking, pn.Spacer(width=40), self.button_clear_roi, self.button_done_roi),
                         self.frame_pane,
                         margin=(10, 0))
                         
                             
def get_tab():
    track = TrackingTab()
    return track.get_panel()
