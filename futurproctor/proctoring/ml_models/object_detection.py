import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize the YOLOv11 model
model = YOLO("yolo11s.pt")  # Replace with your YOLOv11 model file

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Define allowed classes
ALLOWED_CLASSES = {"cell phone", "book"}

def detectObject(frame, confidence_threshold=CONFIDENCE_THRESHOLD, resize_width=640):
    """
    Perform object detection on a single frame, filtering only "cell phone" and "book".
    
    Args:
        frame (ndarray): Input image frame in BGR format.
        confidence_threshold (float): Confidence threshold for object detection.
        resize_width (int): Width to resize the frame for faster processing. Aspect ratio is maintained.
    
    Returns:
        labels_this_frame (list): List of detected labels with their confidence scores.
        processed_frame (ndarray): Frame with detection results (bounding boxes and labels).
    """
    labels_this_frame = []

    # Validate input frame
    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("Invalid frame. Please provide a valid numpy array.")

    # Resize the frame to improve processing speed
    height, width = frame.shape[:2]
    if width > resize_width:
        aspect_ratio = height / width
        frame = cv2.resize(frame, (resize_width, int(resize_width * aspect_ratio)))

    try:
        # Perform object detection
        results = model(frame)

        for result in results:
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, class_id = box

                if score > confidence_threshold:  # Apply confidence threshold
                    label = model.names[int(class_id)]
                    
                    if label in ALLOWED_CLASSES:  # Filter for allowed classes
                        labels_this_frame.append((label, float(score)))

                        # Draw bounding box in blue
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        # Draw label and confidence value in red
                        cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        logging.info(f"Detected objects: {labels_this_frame}")

    except Exception as e:
        logging.error(f"Error during object detection: {e}")
        raise e

    return labels_this_frame, frame

# # Test the object detection function
# if __name__ == "__main__":
#     # Use a webcam or video feed for testing
#     cap = cv2.VideoCapture(0)  # Change to video file path if needed

#     if not cap.isOpened():
#         logging.error("Failed to access the video feed.")
#         exit(1)

#     logging.info("Starting object detection. Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logging.warning("No frame captured from video feed.")
#             break

#         try:
#             labels, processed_frame = detectObject(frame)
#             cv2.imshow("Object Detection", processed_frame)

#         except Exception as e:
#             logging.error(f"Error in processing frame: {e}")
#             break

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     logging.info("Object detection stopped.")
