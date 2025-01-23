# import cv2
# import mediapipe as mp
# import csv

# def extract_pose(video_file, output_csv):
#     """
#     Extracts pose landmarks from a video and saves them to a CSV file.

#     Args:
#         video_file (str): Path to the input video file.
#         output_csv (str): Path to the output CSV file.

#     Returns:
#         None
#     """
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     cap = cv2.VideoCapture(video_file)

#     frame_count = 0
#     with open(output_csv, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['frame', 'keypoint', 'x', 'y', 'z', 'visibility'])

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_count += 1
#             # Convert the image to RGB
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(rgb_frame)

#             if results.pose_landmarks:
#                 for i, landmark in enumerate(results.pose_landmarks.landmark):
#                     writer.writerow([frame_count, mp_pose.PoseLandmark(i).name, landmark.x, landmark.y, landmark.z, landmark.visibility])

#             # Optional: Display the video feed with pose landmarks
#             # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#             # cv2.imshow('Pose Detection', frame)

#             # Stop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import csv

def extract_pose(video_file, output_csv):
    """
    Extracts hand landmarks from a video and saves them to a CSV file.

    Args:
        video_file (str): Path to the input video file.
        output_csv (str): Path to the output CSV file.

    Returns:
        None
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, 
                           max_num_hands=2, 
                           min_detection_confidence=0.5, 
                           min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_file)

    frame_count = 0
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'hand', 'landmark', 'x', 'y', 'z'])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Convert the image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        writer.writerow([
                            frame_count, 
                            f"hand_{hand_index + 1}",  # 'hand_1' or 'hand_2'
                            mp_hands.HandLandmark(i).name,  # Name of the landmark
                            landmark.x, landmark.y, landmark.z
                        ])

            # Optional: Display the video feed with hand landmarks
            # for hand_landmarks in results.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # cv2.imshow('Hand Detection', frame)

            # Stop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# # Example usage
# video_file = "predicted_videos/concatenated_video.mp4"
# output_csv = "hand_landmarks.csv"
# extract_hand_landmarks(video_file, output_csv)
