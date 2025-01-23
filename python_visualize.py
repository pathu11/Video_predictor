import csv
import pywavefront

# Specify the correct path for your avatar's OBJ file
# obj_file = "avatar.obj"  # Use the converted OBJ file
obj_file = "avatar_3.obj"

# Ensure the file exists before loading
try:
    # Load the OBJ file using pywavefront
    scene = pywavefront.Wavefront(obj_file)
except FileNotFoundError:
    print(f"Error: The OBJ file '{obj_file}' was not found.")
    exit(1)
except Exception as e:
    print(f"Error loading OBJ file: {e}")
    exit(1)

# Process your CSV pose data
csv_file = "predicted_videos/pose_data.csv"
try:
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            frame = int(row['frame']) 
            bone_name = row['keypoint']
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])

            # Find the bone in the avatar's rig and set its position
            for bone in scene.bones:
                if bone.name == bone_name:
                    # Assuming bones have translation properties
                    bone.set_position(x, y, z)
                    bone.insert_keyframe(frame)  # Insert keyframe at the specified frame

except FileNotFoundError:
    print(f"Error: The CSV file '{csv_file}' was not found.")
    exit(1)

# Save the animated avatar as an OBJ file
scene.save("animated_avatar.obj")  # Save as OBJ file
print("Animation exported to animated_avatar.obj")
