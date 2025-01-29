import cv2
import math
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Camera settings
size = (480, 640, 3)
font = cv2.FONT_HERSHEY_PLAIN

# Camera internals
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)
dist_coeffs = np.zeros((4, 1))

# 3D model points
model_points = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
])


def get_head_pose_angles(rotation_vector):
    """Calculate yaw, pitch, and roll from the rotation vector."""
    # Convert the rotation vector to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extract angles
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    pitch = math.atan2(-rotation_matrix[2, 0], sy)  # Up/Down
    yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Left/Right
    roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Tilt

    # Convert radians to degrees
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    roll = math.degrees(roll)

    return pitch, yaw, roll


def head_pose_detection(img):
    """Detect head pose and classify direction."""
    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Get 2D image points from the face landmarks
            image_points = np.array([
                [landmarks.landmark[1].x * img.shape[1], landmarks.landmark[1].y * img.shape[0]],   # Nose tip
                [landmarks.landmark[152].x * img.shape[1], landmarks.landmark[152].y * img.shape[0]],  # Chin
                [landmarks.landmark[33].x * img.shape[1], landmarks.landmark[33].y * img.shape[0]],   # Left eye left corner
                [landmarks.landmark[263].x * img.shape[1], landmarks.landmark[263].y * img.shape[0]], # Right eye right corner
                [landmarks.landmark[287].x * img.shape[1], landmarks.landmark[287].y * img.shape[0]], # Left mouth corner
                [landmarks.landmark[57].x * img.shape[1], landmarks.landmark[57].y * img.shape[0]]    # Right mouth corner
            ], dtype="double")

            # SolvePnP to get the rotation and translation vectors
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            # Get head pose angles
            pitch, yaw, roll = get_head_pose_angles(rotation_vector)

            # Display angles on the screen
            cv2.putText(img, f"Pitch: {pitch:.2f}", (50, 50), font, 1.2, (0, 255, 0), 2)
            cv2.putText(img, f"Yaw: {yaw:.2f}", (50, 80), font, 1.2, (0, 255, 0), 2)
            cv2.putText(img, f"Roll: {roll:.2f}", (50, 110), font, 1.2, (0, 255, 0), 2)

            # Classify direction based on angles
            if pitch > 10:
                cv2.putText(img, 'Looking Up', (50, 150), font, 1.5, (0, 0, 255), 2)
            elif pitch < -10:
                cv2.putText(img, 'Looking Down', (50, 150), font, 1.5, (0, 0, 255), 2)
            
            if yaw > 10:
                cv2.putText(img, 'Looking Right', (50, 180), font, 1.5, (0, 0, 255), 2)
            elif yaw < -10:
                cv2.putText(img, 'Looking Left', (50, 180), font, 1.5, (0, 0, 255), 2)

    return img


# # Example usage with webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = head_pose_detection(frame)

#     cv2.imshow("Head Pose Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
