import json
from datetime import datetime

import bokeh
from bokeh.models import BoxAnnotation, PolyAnnotation, Circle

# debug
from pprint import pprint

def export_roi_data(rois, roi_count, image_height, image_width):
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
            fixed_ys = [image_height-i for i in roi.ys]
            
            json_list.append({
                "type": "polygon",
                "pts": list(zip(map(int, roi.xs), map(int, fixed_ys)))
            })
            
        if str(type(roi)) == "<class 'bokeh.models.renderers.glyph_renderer.GlyphRenderer'>":            
            try:
                # Use saved values if available
                if hasattr(roi, 'actual_radius') and hasattr(roi, 'center_x') and hasattr(roi, 'center_y'):
                    center_x = int(roi.center_x)
                    center_y = int(roi.center_y)
                    radius = int(roi.actual_radius)
                elif 'x' in roi.data_source.data and 'y' in roi.data_source.data:
                    center_x = int(roi.data_source.data['x'][0])
                    center_y = int(roi.data_source.data['y'][0])
                    if 'size' in roi.data_source.data:
                        radius = int(roi.data_source.data['size'][0] / 2)
                    else:
                        radius = 10
                else:
                    # Fallback to old format
                    center_x = int(roi.glyph.x)
                    center_y = int(roi.glyph.y)
                    radius = int(roi.glyph.radius)
                    
                json_list.append({
                    "type": "circle",
                    "center": (center_x, image_height - center_y), 
                    "radius": radius
                })
            except (AttributeError, KeyError, ValueError, TypeError) as e:
                print(f"Error exporting circle ROI: {e}")
            
    json_dict["rois"] = json_list
    with open("rois.json", "w") as file:
        json.dump(json_dict, file, indent=2)