import csv
from collections import defaultdict

# Define a dictionary to store hand pose data (frame -> hand_name -> landmarks)
pose_data = defaultdict(lambda: defaultdict(list))

# Load the CSV file with hand pose data
csv_file = "predicted_videos/pose_data.csv"  # Adjust to your file path
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        frame = int(row['frame'])
        hand_name = row['hand']
        landmark_name = row['landmark']
        x = float(row['x'])
        y = float(row['y'])
        z = float(row['z'])
        
        # Store the data in the dictionary
        pose_data[frame][hand_name].append((landmark_name, x, y, z))

# Save the pose data to a JSON file (for easier loading in Three.js)
import json
with open('pose_data.json', 'w') as f:
    json.dump(pose_data, f)
