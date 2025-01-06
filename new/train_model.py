import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
from preprocess_input import get_text_embedding, get_video_feature
import cv2  
DATA_DIR = './processed_dataset/train/'

def extract_first_frame(video_file_path):
  
    cap = cv2.VideoCapture(video_file_path)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_file_path}")
    
    ret, frame = cap.read()
    if not ret:
        raise Exception(f"Could not read frame from video file: {video_file_path}")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    cap.release()  
    
    return frame
def get_video_files(word_folder):
    """Find all video files in the given folder"""
    video_files = []
    video_extensions = ['.mp4', '.mov']
    for filename in os.listdir(word_folder):
        if any(filename.endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(word_folder, filename))
    return video_files

def generate_data():
    """Extract features and store for training"""
    word_embeddings = []
    video_embeddings = []
    word_to_video_map = {}

    # Iterate through all categories and words
    for category in os.listdir(DATA_DIR):
        category_path = os.path.join(DATA_DIR, category)
        category_path = os.path.normpath(category_path) 
        
        for word in os.listdir(category_path):
            word_path = os.path.join(category_path, word)
            word_path = os.path.normpath(word_path)  
            
            word_embedding = get_text_embedding(word)
            word_embeddings.append(word_embedding)

            video_files = get_video_files(word_path)
            
            if video_files:
                word_video_embeddings = []
                for video_file_path in video_files:
                    try:
                      
                        video_frame = extract_first_frame(video_file_path)
                        video_embedding = get_video_feature(video_frame)
                        word_video_embeddings.append(video_embedding)
                    except Exception as e:
                        print(f"Warning: Could not process {video_file_path} for word '{word}'. Error: {e}")
                
                if word_video_embeddings:
                
                    aggregated_video_embedding = np.mean(word_video_embeddings, axis=0)
                    video_embeddings.append(aggregated_video_embedding)
                    word_to_video_map[word] = video_files 
            else:
                print(f"Warning: No video files found for word '{word}' in folder {word_path}. Skipping this word.")

    return np.array(word_embeddings), np.array(video_embeddings), word_to_video_map

def train_model():
    """Train a Nearest Neighbors model to match text to video"""
    word_embeddings, video_embeddings, word_to_video_map = generate_data()

    model = NearestNeighbors(n_neighbors=1, metric='cosine')
    model.fit(word_embeddings, video_embeddings)

    with open('./models/word_embeddings.pkl', 'wb') as f:
        pickle.dump(word_embeddings, f)
    with open('./models/video_embeddings.pkl', 'wb') as f:
        pickle.dump(video_embeddings, f)
    with open('./models/word_to_video_map.pkl', 'wb') as f:
        pickle.dump(word_to_video_map, f)
    with open('./models/nearest_neighbors_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model()
