import json
from datetime import datetime

import bokeh
from bokeh.models import BoxAnnotation, PolyAnnotation
from pprint import pprint

def export_tracking_data(rois, image_height, image_width):
    json_list = []
    
    for roi in rois: 
        if str(type(roi)) == "<class 'bokeh.models.annotations.geometry.BoxAnnotation'>":
            print("Box")
            # pprint(vars(roi))
            
            json_list.append({
                "top": int(roi.top),
                "bottom": int(roi.bottom),
                "right": int(roi.right),
                "left": int(roi.left)
            })
                        
        # if str(type(roi)) == "<class 'bokeh.models.annotations.geometry.PolyAnnotation'>":
        #     print("Poly")
        #     pprint(vars(roi))
            
        # if str(type(roi)) == "<class 'bokeh.models.renderers.glyph_renderer.GlyphRenderer'>":
        #     print("Circle")
        #     pprint(vars(roi))
        
    print(json_list)
    with open("rois.json", "w") as file:
        json.dump(json_list, file, indent=2)