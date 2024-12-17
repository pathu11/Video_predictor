import cv2
import json
from openpose import pyopenpose as op

# Path to OpenPose model files
params = {
    "model_folder": "D:/3rd year/Internship/datasets/openpose/models/",  # Example: "models/"
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Load video
video_path = "predicted_videos/concatenated_video.avi"
cap = cv2.VideoCapture(video_path)

# Store keypoints for each frame
pose_data = []

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # Extract keypoints
    keypoints = []
    if datum.poseKeypoints is not None:
        for person in datum.poseKeypoints:  # Loop through all detected people
            person_keypoints = []
            for keypoint in person:  # Loop through each body landmark
                person_keypoints.append({
                    "x": float(keypoint[0]),
                    "y": float(keypoint[1]),
                    "confidence": float(keypoint[2])
                })
            keypoints.append(person_keypoints)

        pose_data.append({"frame": frame_count, "keypoints": keypoints})

    # Visualize (Optional)
    cv2.imshow("Pose Detection", datum.cvOutputData)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save keypoints to a JSON file
with open("pose_data.json", "w") as f:
    json.dump(pose_data, f)
