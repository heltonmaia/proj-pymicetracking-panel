import json
from datetime import datetime

import bokeh
from bokeh.models import BoxAnnotation, PolyAnnotation, Circle

# debug
from pprint import pprint

def export_tracking_data(rois, roi_count, image_height, image_width):
    json_list = []
    
    json_dict = {
        "rois": [],
        "roi_counts": roi_count
    }
    
    for roi in rois:
        # image_height - height_in frame happens bc of orientation in xy-axis
        
        if str(type(roi)) == "<class 'bokeh.models.annotations.geometry.BoxAnnotation'>":
            # validation to avoid negative numbers
            if roi.top < 0:
                roi.top = 0
            if roi.bottom < 0:
                roi.bottom = 0
            if roi.right < 0:
                roi.right = 0
            if roi.left < 0:
                roi.left = 0

            json_list.append({
                "type": "box",
                "top":  image_height-int(roi.top),
                "bottom": image_height-int(roi.bottom),
                "right": int(roi.right),
                "left":  int(roi.left)
            })
                        
        if str(type(roi)) == "<class 'bokeh.models.annotations.geometry.PolyAnnotation'>":           
            print(roi.ys)
            fixed_ys = [image_height-i for i in roi.ys]
            print(fixed_ys)
            
            json_list.append({
                "type": "polygon",
                "pts": list(zip(map(int, roi.xs), map(int, fixed_ys)))
            })
            
        if str(type(roi)) == "<class 'bokeh.models.renderers.glyph_renderer.GlyphRenderer'>":
            print("Circle")
            
            json_list.append({
                "type": "circle",
                "center": (int(roi.glyph.x), image_height-int(roi.glyph.y)), 
                "radius": int(roi.glyph.radius)
            })
            
    json_dict["rois"] = json_list
    with open("rois.json", "w") as file:
        json.dump(json_dict, file, indent=2)