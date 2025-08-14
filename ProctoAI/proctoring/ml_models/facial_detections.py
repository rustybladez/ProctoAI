import cv2
import mediapipe as mp
import numpy as np
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize face detection and face mesh models
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

"""NEW FEATURE FOR BRIGHTNESS"""

def _safe_bbox_to_int(bbox_rel, image_shape):
    """Convert relative bbox from MediaPipe to integer pixel bbox (x1,y1,x2,y2), clamped."""
    ih, iw = image_shape[:2]
    xmin = int(max(0, bbox_rel.xmin * iw))
    ymin = int(max(0, bbox_rel.ymin * ih))
    w = int(bbox_rel.width * iw)
    h = int(bbox_rel.height * ih)
    x2 = min(iw, xmin + max(1, w))
    y2 = min(ih, ymin + max(1, h))
    return xmin, ymin, x2, y2

def _compute_brightness_stats(frame):
    """Return mean and variance of grayscale image (frame)."""
    if frame is None:
        return None, None
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)), float(np.var(gray))
    except Exception as e:
        logger.error(f"Brightness compute error: {e}")
        return None, None

def detectFace(frame):
    """
    Detect faces and compute brightness statistics.

    Returns:
      face_count (int),
      annotated_frame (BGR numpy array),
      brightness (dict) with keys:
        - frame_mean, frame_var
        - face_mean, face_var (None if no face detected)
    """
    if frame is None:
        return 0, None, {'frame_mean': None, 'frame_var': None, 'face_mean': None, 'face_var': None}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_frame = frame.copy()

    # Whole-frame brightness
    frame_mean, frame_var = _compute_brightness_stats(annotated_frame)

    # Run MediaPipe face detection
    try:
        detection_results = face_detection.process(rgb_frame)
    except Exception as e:
        logger.error(f"MediaPipe face detection error: {e}")
        detection_results = None

    face_count = 0
    face_mean = None
    face_var = None

    if detection_results and detection_results.detections:
        face_count = len(detection_results.detections)
        for det in detection_results.detections:
            # Draw detection box/landmarks
            try:
                mp_drawing.draw_detection(annotated_frame, det)
            except Exception:
                pass

            # compute face ROI brightness for the first detection only
            if face_mean is None:
                bbox_rel = det.location_data.relative_bounding_box
                x1, y1, x2, y2 = _safe_bbox_to_int(bbox_rel, annotated_frame.shape)
                # crop with safety checks
                try:
                    roi = annotated_frame[y1:y2, x1:x2]
                    if roi.size != 0:
                        face_mean, face_var = _compute_brightness_stats(roi)
                except Exception as e:
                    logger.debug(f"Error cropping face ROI: {e}")
                    face_mean, face_var = None, None

    # Also process face mesh landmarks for richer annotation if needed
    try:
        mesh_results = face_mesh.process(rgb_frame)
        if mesh_results and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )
    except Exception:
        pass

    brightness = {
        'frame_mean': frame_mean,
        'frame_var': frame_var,
        'face_mean': face_mean,
        'face_var': face_var
    }

    return face_count, annotated_frame, brightness
# ...existing code...
    
"""NEW FEATURE FOR BRIGHTNESS END"""

# def detectFace(frame):
#     """
#     Detects faces, landmarks, and alerts on suspicious activities (e.g., multiple faces or suspicious gaze).
#     Returns: faceCount, annotated frame
#     """
#     # Convert the frame to RGB as required by MediaPipe
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faceCount = 0

#     # Detect faces in the frame
#     detection_results = face_detection.process(rgb_frame)
#     annotated_frame = frame.copy()

#     if detection_results.detections:
#         faceCount = len(detection_results.detections)

#         # Draw bounding boxes and landmarks
#         for detection in detection_results.detections:
#             mp_drawing.draw_detection(annotated_frame, detection)

#     # Alert for multiple faces
#     if faceCount > 1:
#         cv2.putText(annotated_frame, 'Alert: Multiple Faces Detected!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Detect facial landmarks using Face Mesh
#     mesh_results = face_mesh.process(rgb_frame)
#     if mesh_results.multi_face_landmarks:
#         for face_landmarks in mesh_results.multi_face_landmarks:
#             # Draw the facial landmarks on the frame
#             mp_drawing.draw_landmarks(
#                 image=annotated_frame,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=None,
#                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
#             )

#     return faceCount, annotated_frame
