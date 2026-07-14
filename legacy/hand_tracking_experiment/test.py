import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup MediaPipe Task Options
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # Required for webcam/video streams
    num_hands=2
)

# 2. Initialize the OpenCV Webcam Capture
cap = cv2.VideoCapture(0)

# 3. Create the Detector using a Context Manager
with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # OpenCV reads BGR; MediaPipe requires RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the OpenCV frame to a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate current timestamp in milliseconds (required for VIDEO mode)
        frame_timestamp_ms = int(time.time() * 1000)

        # Perform hand detection
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        # 4. Draw the Landmark Points manually on the frame
        if detection_result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    # Scale normalized coordinates back to actual pixel locations
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # Display the output
        cv2.imshow('MediaPipe Tasks Hand Tracking', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()