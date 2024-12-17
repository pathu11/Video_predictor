import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Keypoint names for reference
keypoint_names = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

# Open video
video_path = "test.mp4"  # Path to pose-estimated video
cap = cv2.VideoCapture(video_path)

# Open CSV file for writing
output_csv_path = "pose_data.csv"
with open(output_csv_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header
    csv_writer.writerow(["frame", "keypoint", "x", "y", "z", "visibility"])

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for pose estimation
        result = pose.process(rgb_frame)

        # Extract landmarks if detected
        if result.pose_landmarks:
            for i, landmark in enumerate(result.pose_landmarks.landmark):
                csv_writer.writerow([
                    frame_count,          # Current frame number
                    keypoint_names[i],    # Keypoint name
                    landmark.x,           # X coordinate
                    landmark.y,           # Y coordinate
                    landmark.z,           # Z coordinate
                    landmark.visibility   # Visibility score
                ])
            print(f"Frame {frame_count}: Pose data written to CSV.")
        else:
            print(f"Frame {frame_count}: No landmarks detected.")
        
        frame_count += 1

cap.release()
print(f"Pose data saved to {output_csv_path}")
