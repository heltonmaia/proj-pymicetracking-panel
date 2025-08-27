"""
Frame processing and tracking functions
"""

import traceback

import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO

from shapely.geometry import Point, Polygon
from .detection import calculate_centroid, template_matching

COLORS = {
    "mask": (255, 165, 0),  # Orange (YOLO detection)
    "centroid": (0, 255, 255),  # Yellow (YOLO centroid)
    "template_mask": (0, 255, 0),  # Green (Template detection)
    "template_centroid": (255, 0, 255),  # Purple (Template centroid)
    "text": (255, 255, 255),  # White
    "roi_rectangle": (0, 0, 105), # Dark red
    "roi_circle": (0, 255, 0),
    "roi_polygon": (100,0,0),
}

def get_current_roi(params, rois: list, frame_height, centroid_x, centroid_y):
    """
    
    """
    centroid = Point(centroid_x, centroid_y)
    print(rois)
    for index, roi in enumerate(rois): 
        if str(type(roi)) == "<class 'bokeh.models.annotations.geometry.BoxAnnotation'>":
            x0, y0, x1, y1 = list(map(int, [roi.left, frame_height-roi.top, roi.right, frame_height-roi.bottom]))
            rectangle = Polygon([(x0, y0), (x0, y1), (x1, y0), (x1, y1)])
            
            if rectangle.contains(centroid):
                print("rect in track .py")
                params.frame_per_roi[index] += 1
            
        elif str(type(roi)) == "<class 'bokeh.models.renderers.glyph_renderer.GlyphRenderer'>":  
            center = Point(int(roi.glyph.x), frame_height-int(roi.glyph.y))
            circle = center.buffer(int(roi.glyph.radius))
    
            if circle.contains(centroid):
                params.frame_per_roi[index] += 1

        elif str(type(roi)) == "<class 'bokeh.models.annotations.geometry.PolyAnnotation'>":
            fixed_ys = [frame_height-i for i in roi.ys]
            
            pts = np.array(list(zip(map(int, roi.xs), map(int, fixed_ys))), np.int32)
                
    
def create_roi_mask(rois: list, frame_shape: tuple[int, int]) -> np.ndarray:
    """
    Cria uma máscara binária a partir dos ROIs definidos.
    Args:
        rois: lista de objetos ROI (Rectangle ou Circle)
        frame_shape: shape do frame em cinza (altura, largura)
    Returns:
        np.ndarray: máscara binária (uint8)
    """
    mask = np.zeros(frame_shape, dtype=np.uint8)

    height, width = frame_shape

    for roi in rois:
        if str(type(roi)) == "<class 'bokeh.models.annotations.geometry.BoxAnnotation'>":
            x0, y0, x1, y1 = list(map(int, [roi.left, height-roi.top, roi.right, height-roi.bottom]))
            
            cv.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

        elif str(type(roi)) == "<class 'bokeh.models.renderers.glyph_renderer.GlyphRenderer'>":            
            pass
            # cv.circle(mask, (int(roi.glyph.x), height-int(roi.glyph.y)), int(roi.glyph.radius), 255, -1)
        
        elif str(type(roi)) == "<class 'bokeh.models.annotations.geometry.PolyAnnotation'>":
            fixed_ys = [height-i for i in roi.ys]
            
            pts = np.array(list(zip(map(int, roi.xs), map(int, fixed_ys))), np.int32)
            pts = pts.reshape((-1,1,2))
            cv.polylines(mask, [pts], True, (0,255,255))

    return mask

def draw_rois(image: np.ndarray, rois: list):
    height, _, _ = image.shape

    for roi in rois:
        if str(type(roi)) == "<class 'bokeh.models.annotations.geometry.BoxAnnotation'>":
            x0, y0, x1, y1 = list(map(int, [roi.left, height-roi.top, roi.right, height-roi.bottom]))
            cv.rectangle(image, (x0, y0), (x1, y1), COLORS["roi_rectangle"], 2)

        elif str(type(roi)) == "<class 'bokeh.models.renderers.glyph_renderer.GlyphRenderer'>":  
            cv.circle(image, (int(roi.glyph.x), height-int(roi.glyph.y)), int(roi.glyph.radius), COLORS["roi_circle"], 2)
    
        elif str(type(roi)) == "<class 'bokeh.models.annotations.geometry.PolyAnnotation'>":
            fixed_ys = [height-i for i in roi.ys]
        
            pts = np.array(list(zip(map(int, roi.xs), map(int, fixed_ys))), np.int32)
            pts = pts.reshape((-1,1,2))
            cv.polylines(image, [pts], True, COLORS["roi_polygon"], 2)

    return image

def process_frame(frame: np.ndarray, model: YOLO, frame_num: int, params) -> np.ndarray:
    """
    Process a single frame for mouse detection and ROI tracking.

    Args:
        frame: Input frame (RGB format)
        model: YOLO model for detection
        frame_num: Frame number for tracking

    Returns:
        np.ndarray: Processed frame with overlays
    """
    try:
        overlay = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        centroid = None
        current_roi = None
        detection_made = False

        # Initialize tracking data for this frame
        frame_data = {
            "frame_number": frame_num,
            "centroid_x": None,
            "centroid_y": None,
            "roi": None,
            "detection_method": None,
        }

        # First try YOLO detection
        with torch.no_grad():
            results = model(
                frame,
                verbose=False,
                conf=params.slider_confidence.value,
                iou=params.slider_iou.value,
                device=params.device,
                imgsz=640,
            )

        if results and len(results) > 0:
            if (
                hasattr(results[0], "masks")
                and results[0].masks is not None
                and len(results[0].masks) > 0
            ):
                best_idx = np.argmax(results[0].boxes.conf.cpu().numpy())

                if best_idx < len(results[0].masks.data):
                    best_mask_tensor = results[0].masks.data[best_idx].cpu()

                    best_mask = cv.resize(
                        best_mask_tensor.numpy(),
                        (frame_width, frame_height),
                        interpolation=cv.INTER_NEAREST,
                    )

                    binary_mask = (best_mask > 0.5).astype(np.uint8)
                    centroid = calculate_centroid(binary_mask)

                    if centroid:
                        detection_made = True
                        params.yolo_detections += 1

                        frame_data.update(
                            {
                                "centroid_x": centroid[0],
                                "centroid_y": centroid[1],
                                "detection_method": "YOLO",
                            }
                        )

                        colored_mask = np.zeros_like(overlay)
                        if (
                            binary_mask.shape[0] == colored_mask.shape[0]
                            and binary_mask.shape[1] == colored_mask.shape[1]
                        ):
                            colored_mask[binary_mask > 0] = COLORS["mask"]
                            overlay = cv.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
                            cv.circle(overlay, centroid, 4, COLORS["centroid"], -1)

        # If YOLO didn't detect anything, try template matching
        if not detection_made and params.background_frame is not None:
            current_frame_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            # Gere a máscara dos ROIs
            roi_mask = create_roi_mask(params.rois, current_frame_gray.shape)
            centroid = template_matching(
                current_frame_gray, params.background_frame, roi_mask
            )

            if centroid:
                detection_made = True
                params.template_detections += 1

                frame_data.update(
                    {
                        "centroid_x": centroid[0],
                        "centroid_y": centroid[1],
                        "detection_method": "Template",
                    }
                )

                # Draw template detection with different color
                cv.circle(overlay, centroid, 8, COLORS["template_centroid"], 2)
                cv.circle(overlay, centroid, 4, COLORS["template_centroid"], -1)

        # Check if centroid is in any ROI
        # print(f"Centroid: {centroid}")
        if centroid:            
            get_current_roi(params, params.roi_in_track, frame_height, frame_data["centroid_x"], frame_data["centroid_y"])
            # frame_data["roi"] = current_roi
            # params.frame_per_roi[current_roi] += 1
            
        # Update no detection counter
        if not detection_made:
            params.no_detection_count += 1
            frame_data["detection_method"] = "None"

        # Add frame data to tracking data
        params.tracking_data.append(frame_data)
        params.frame_count += 1

        return overlay

    except Exception as e: # consertar para mostrar aviso na tela
        print(f"Frame {frame_num} error: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        print(traceback.format_exc())

        return frame
