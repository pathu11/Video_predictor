import cv2
import mediapipe as mp
import json

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load video
video_path = "test.mkv"
cap = cv2.VideoCapture(video_path)

# Check if the video is loaded successfully
if not cap.isOpened():
    print(f"Error: Unable to open video {video_path}")
    exit()

# Store keypoints for each frame
pose_data = []

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect pose
    result = pose.process(rgb_frame)

    # Extract keypoints
    if result.pose_landmarks:
        keypoints = []
        for landmark in result.pose_landmarks.landmark:
            keypoints.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
        pose_data.append({"frame": frame_count, "keypoints": keypoints})
        print(f"Frame {frame_count}: Keypoints detected.")
    else:
        print(f"Frame {frame_count}: No landmarks detected.")

    # Visualize (Optional)
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("Pose Detection", frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save keypoints to a JSON file
if pose_data:
    with open("pose_data.json", "w") as f:
        json.dump(pose_data, f)
    print("Pose data saved to pose_data.json")
else:
    print("No pose data to save.")
