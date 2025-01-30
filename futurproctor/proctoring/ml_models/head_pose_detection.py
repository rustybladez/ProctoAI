import cv2
import math
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def get_head_pose_angles(rotation_vector):
    """Calculate yaw, pitch, and roll from the rotation vector."""
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    pitch = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
    yaw = math.degrees(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    roll = math.degrees(math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
    return {"pitch": pitch, "yaw": yaw, "roll": roll}

def head_pose_detection(frame):
    """Detect head pose and return angles (pitch, yaw, roll)."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Extract key points for head pose estimation
            image_points = np.array([
                [landmarks.landmark[1].x * frame.shape[1], landmarks.landmark[1].y * frame.shape[0]],   # Nose tip
                [landmarks.landmark[152].x * frame.shape[1], landmarks.landmark[152].y * frame.shape[0]],  # Chin
                [landmarks.landmark[33].x * frame.shape[1], landmarks.landmark[33].y * frame.shape[0]],   # Left eye left corner
                [landmarks.landmark[263].x * frame.shape[1], landmarks.landmark[263].y * frame.shape[0]], # Right eye right corner
                [landmarks.landmark[287].x * frame.shape[1], landmarks.landmark[287].y * frame.shape[0]], # Left mouth corner
                [landmarks.landmark[57].x * frame.shape[1], landmarks.landmark[57].y * frame.shape[0]]    # Right mouth corner
            ], dtype="double")

            # 3D model points
            model_points = np.array([
                (0.0, 0.0, 0.0),            # Nose tip
                (0.0, -330.0, -65.0),       # Chin
                (-225.0, 170.0, -135.0),    # Left eye left corner
                (225.0, 170.0, -135.0),     # Right eye right corner
                (-150.0, -150.0, -125.0),   # Left mouth corner
                (150.0, -150.0, -125.0)     # Right mouth corner
            ])

            # Camera matrix
            focal_length = frame.shape[1]
            center = (frame.shape[1] / 2, frame.shape[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            # SolvePnP to get rotation vector
            dist_coeffs = np.zeros((4, 1))
            success, rotation_vector, _ = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            if success:
                return get_head_pose_angles(rotation_vector)
    
    # Return default angles if no face detected
    return {"pitch": 0, "yaw": 0, "roll": 0}