import json
from datetime import datetime
from typing import Any


def _convert_rois_to_dict(rois: list) -> list[dict]:
    """Convert ROI objects to dictionary format"""
    result = []

    for roi in rois:
        try:
            if hasattr(roi, "to_dict"):
                result.append(roi.to_dict())
            elif hasattr(roi, "__dict__"):
                result.append(roi.__dict__)
            else:
                result.append(str(roi))
        except Exception as e:
            print(f"Warning: Could not convert ROI to dict: {e}")
            result.append({"error": f"Could not convert ROI: {str(e)}"})

    return result


def export_tracking_data(
    video_name: str,
    experiment_type: str,
    total_frames: int,
    frames_without_detection: int,
    yolo_detections: int,
    template_detections: int,
    rois: list,
    roi_counts: dict[int, int],
    tracking_data: list[dict[str, Any]],
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
        "rois": _convert_rois_to_dict(rois),
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
