import open3d as o3d
import pandas as pd
import numpy as np
import time

# Load pose data
df = pd.read_csv("pose_data.csv")
print(df.head())  # Inspect the first few rows of the dataframe
print(df["frame"].unique())  # List all unique frames in the data

# Define body connections (pairs of keypoints)
BODY_CONNECTIONS = [
    ("NOSE", "LEFT_SHOULDER"),
    ("NOSE", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    # Add more body connections as needed
]

# Create a dictionary to store 3D points for a specific frame
def get_frame_points(frame_number, visibility_threshold=0.5):
    frame_data = df[df["frame"] == frame_number]
    if frame_data.empty:
        print(f"No data found for frame {frame_number}")
        return {}
    
    keypoints = {
        row["keypoint"]: [row["x"], row["y"], row["z"]]
        for _, row in frame_data.iterrows()
        if row["visibility"] >= visibility_threshold
    }

    if not keypoints:
        print(f"No visible keypoints found for frame {frame_number}")
    
    return keypoints


# Create lines and points for visualization
def create_skeleton(points):
    line_set = o3d.geometry.LineSet()
    point_cloud = o3d.geometry.PointCloud()

    # Create point cloud from keypoints
    point_list = list(points.values())
    point_cloud.points = o3d.utility.Vector3dVector(np.array(point_list))

    # Create connections (lines) between keypoints
    connections = []
    for start, end in BODY_CONNECTIONS:
        if start in points and end in points:
            start_idx = list(points.keys()).index(start)
            end_idx = list(points.keys()).index(end)
            connections.append([start_idx, end_idx])

    line_set.points = o3d.utility.Vector3dVector(np.array(point_list))
    line_set.lines = o3d.utility.Vector2iVector(np.array(connections))

    return point_cloud, line_set

# Visualize the skeleton frame by frame
def visualize_skeleton():
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    point_cloud = o3d.geometry.PointCloud()
    line_set = o3d.geometry.LineSet()

    vis.add_geometry(point_cloud)
    vis.add_geometry(line_set)

    max_frame = df["frame"].max()

    for frame_number in range(1, max_frame + 1):
        points = get_frame_points(frame_number)
        if not points:
            continue

        new_point_cloud, new_line_set = create_skeleton(points)

        # Update geometries
        point_cloud.points = new_point_cloud.points
        line_set.points = new_line_set.points
        line_set.lines = new_line_set.lines

        vis.update_geometry(point_cloud)
        vis.update_geometry(line_set)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.05)  # Adjust for smoother playback

    vis.destroy_window()

if __name__ == "__main__":
    visualize_skeleton()
