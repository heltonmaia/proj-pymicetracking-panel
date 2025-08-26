"""
Data export utilities
"""

import json
from datetime import datetime
from typing import Any, Dict, List

def export_tracking_data(
    video_name: str,
    experiment_type: str,
    total_frames: int,
    frames_without_detection: int,
    yolo_detections: int,
    template_detections: int,
    rois: List,
    roi_counts: Dict[int, int],
    tracking_data: List[Dict[str, Any]],
) -> bytes:
    """
    Export tracking data to JSON format.

    Args:
        - video_name: Name of the processed video
        - experiment_type: Type of experiment
        - total_frames: Total number of frames processed
        - frames_without_detection: Number of frames without detection
        - yolo_detections: Number of YOLO detections
        - template_detections: Number of template matching detections
        - rois: List of ROI objects
        - roi_counts: Dictionary of ROI counts
        - tracking_data: List of frame tracking data

    Returns:
        bytes: JSON data as bytes
    """

    export_data = {
        "video_name": video_name,
        "experiment_type": experiment_type,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "total_frames": total_frames,
        "frames_without_detection": frames_without_detection,
        "yolo_detections": yolo_detections,
        "template_detections": template_detections,
        "rois": [roi.to_dict() for roi in rois],
        "roi_counts": roi_counts,
        "tracking_data": tracking_data,
    }

    # Convert to JSON string and then to bytes
    json_string = json.dumps(export_data, indent=2)

    return json_string.encode("utf-8")


def create_download_filename(video_name: str) -> str:
    """
    Create a standardized filename for downloads.

    Args:
        video_name: Original video filename

    Returns:
        str: Formatted filename for download
    """
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    return f"tracking_data_{video_name}_{timestamp}.json"