import torch
import json
import os
import cv2
import csv
import mediapipe as mp
from tokenizer import preprocess_to_isl
from train import VideoPredictionModel
from train import VideoDataset
import nltk
from nltk.corpus import wordnet
from extract_pose import extract_pose
# Ensure WordNet is downloaded
# nltk.download('wordnet')
import numpy as np
import tensorflow as tf

def select_image_for_letter(letter, dataset_folder):
    """
    Select the best image for a letter from its respective folder in the dataset.

    Args:
        letter (str): The letter to select the image for.
        dataset_folder (str): Path to the dataset folder containing images for each letter.

    Returns:
        str: Path to the selected image for the given letter.
    """
    letter_folder = os.path.join(dataset_folder, letter.upper())
    if not os.path.exists(letter_folder):
        print(f"Error: Folder for letter '{letter}' does not exist.")
        return None
    
    images = [os.path.join(letter_folder, f) for f in os.listdir(letter_folder) if f.endswith(('.jpg', '.png'))]
    if not images:
        print(f"Error: No images found for letter '{letter}' in {letter_folder}.")
        return None
    
    # Select the first image for simplicity (you could improve this logic by selecting based on confidence or any other metric)
    return images[0]

def create_fingerspelling_video(word, dataset_folder, output_folder, video_name):
    """
    Create a video for a given word by concatenating images for each letter (fingerspelling).

    Args:
        word (str): The word to convert into a fingerspelling video.
        dataset_folder (str): Directory containing letter image folders.
        output_folder (str): Directory to save the generated video.
        video_name (str): Name of the output video file.

    Returns:
        str: Path to the generated video file.
    """
    video_path = os.path.join(output_folder, video_name)

    # Initialize video writer
    frame_width, frame_height = None, None
    fps = 1  # Display each letter's image for 1 second
    frames = []

    for letter in word:
        if letter.isalpha():  # Only process alphabetic characters
            image_path = select_image_for_letter(letter, dataset_folder)
            if image_path:
                frame = cv2.imread(image_path)
                if frame is not None:
                    frames.append(frame)
                    if frame_width is None or frame_height is None:
                        frame_height, frame_width, _ = frame.shape
                else:
                    print(f"Warning: Could not read image for letter '{letter}' at {image_path}")
            else:
                print(f"Warning: No image found for letter '{letter}'")

    # Create the video file from the frames
    if frames:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Fingerspelling video for '{word}' saved to {video_path}")
    else:
        print(f"Error: No frames generated for the word '{word}'")
        return None

    return video_path

def load_model(model_path, vocab_size, num_categories):
    """
    Load the trained model from the specified path.

    Args:
        model_path (str): Path to the saved model.
        vocab_size (int): Size of the vocabulary.
        num_categories (int): Number of categories for classification.

    Returns:
        VideoPredictionModel: The loaded model in evaluation mode.
    """
    model = VideoPredictionModel(vocab_size, num_categories)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_video(output_folder, video_name, video_data):
    """
    Save video frames to the specified folder.

    Args:
        output_folder (str): Directory to save the video.
        video_name (str): Name of the video file.
        video_data (list): List of frames (each frame is a numpy array).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, video_name)

    if video_data:
        height, width, _ = video_data[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in video_data:
            out.write(frame)
        out.release()
        print(f"Video saved to {output_path}")
    else:
        print("No video frames to save.")

def get_synonyms(word):
    """
    Retrieve synonyms for a given word using WordNet.
    
    Args:
        word (str): The word to find synonyms for.
    
    Returns:
        list: A list of synonyms for the given word.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().capitalize())
    return list(synonyms)

def predict_video_for_words(isl_sentence, word_to_idx, category_to_idx, model, dataset, dataset_f, output_folder):
    """
    Predict the video for each word in the ISL sentence, considering synonyms if the word is not found.
    If no mapping or synonym is found, use fingerspelling for the word.

    Args:
        isl_sentence (str): The sentence in ISL format.
        word_to_idx (dict): Mapping of words to indices.
        category_to_idx (dict): Mapping of categories to indices.
        model (VideoPredictionModel): The trained video prediction model.
        dataset (VideoDataset): Dataset containing video metadata.
        dataset_f (str): Directory containing letter image folders for fingerspelling.
        output_folder (str): Directory to save the predicted videos.

    Returns:
        list: List of video file paths in the order of the ISL tokens.
    """
    tokens = isl_sentence.split()
    token_indices = []
    ordered_videos = []
    video_clips = []

    for idx, token in enumerate(tokens):
        if token in word_to_idx:
            token_indices.append(word_to_idx[token])
        else:
            print(f"Word '{token}' not found, searching for synonyms...")
            synonyms = get_synonyms(token)
            found_synonym = False

            for synonym in synonyms:
                if synonym in word_to_idx:
                    token_indices.append(word_to_idx[synonym])
                    found_synonym = True
                    print(f"Synonym '{synonym}' found for '{token}'.")
                    break

            if not found_synonym:
                print(f"No mapping found for '{token}'. Using fingerspelling...")
                token_indices.append(token)  # Keep the word itself to create a fingerspelling video later.

    for idx, token_idx in enumerate(token_indices):
        if isinstance(token_idx, str):  # Handle fingerspelling for words not in `word_to_idx`
            fingerspelling_video_name = f"predicted_video_{idx}.avi"
            fingerspelling_video_path = create_fingerspelling_video(
                token_idx, dataset_f, output_folder, fingerspelling_video_name
            )
            if fingerspelling_video_path:
                ordered_videos.append(fingerspelling_video_path)
                # video_clips.append(VideoFileClip(fingerspelling_video_path))
        else:
            token_tensor = torch.tensor([token_idx], dtype=torch.long)
            with torch.no_grad():
                category_out, video_out = model(token_tensor)
                category_idx = category_out.argmax(dim=1).item()
                word = list(word_to_idx.keys())[list(word_to_idx.values()).index(token_idx)]
                category = dataset.metadata.get(word, {}).get("category", None)

                if category:
                    word_metadata = dataset.metadata.get(word, {})
                    if 'videos' in word_metadata:
                        video_filename = word_metadata['videos'][0]
                        predicted_video_path = os.path.join(f"dataset/{category}/{word}", video_filename)
                        video_data = cv2.VideoCapture(predicted_video_path)
                        frames = []
                        while True:
                            ret, frame = video_data.read()
                            if not ret:
                                break
                            frames.append(frame)
                        video_data.release()

                        output_video_name = f"predicted_video_{idx}.avi"
                        save_video(output_folder, output_video_name, frames)
                        final_path = os.path.join(output_folder, output_video_name)
                        ordered_videos.append(final_path)
                        # video_clips.append(VideoFileClip(final_path))
                    else:
                        print(f"Warning: No videos found for {category} - {word}")
                else:
                    print(f"Warning: No category found for word: {word}")
    
    return ordered_videos


def clear_previous_videos(output_folder):
    """
    Remove all video files from the output folder.

    Args:
        output_folder (str): Directory containing the videos to be deleted.
    """
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed previous video: {filename}")
        except Exception as e:
            print(f"Error removing {filename}: {e}")

def concat_videos(video_paths, output_path):
    """
    Concatenate a list of videos and save the result to a specified output file.

    Args:
        video_paths (list): List of video file paths to be concatenated.
        output_path (str): The output file path where the concatenated video will be saved.

    Returns:
        None
    """
    
    if not video_paths:
        print("Error: No video files provided for concatenation.")
        return

    cap = cv2.VideoCapture(video_paths[0])
    if not cap.isOpened():
        print(f"Error: Could not open video {video_paths[0]}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release() 

    out.release() 
    print(f"Concatenated video saved to {output_path}")

def convert_to_mp4(input_video_path, output_video_path):
    """
    Convert a video from AVI format to MP4 format.

    Args:
        input_video_path (str): Path to the input AVI video.
        output_video_path (str): Path to the output MP4 video.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Converted video saved to {output_video_path}")



def main():
    """
    Main function to process the ISL sentence, predict videos, and save them in correct order.
    """
    input_text = input("Enter tokenized sentence: ")
    isl_sentence = preprocess_to_isl(input_text)
    print("Processed ISL Sentence:", isl_sentence)

    output_folder = "predicted_videos"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Clear previous videos
    clear_previous_videos(output_folder)

    # Load word and category mappings
    with open("models/word_to_idx.json", "r") as f:
        word_to_idx = json.load(f)

    with open("models/category_to_idx.json", "r") as f:
        category_to_idx = json.load(f)

    model_path = "models/video_prediction_model.pth"
    vocab_size = len(word_to_idx)
    num_categories = len(category_to_idx)

    dataset = VideoDataset(metadata_file="processed_dataset/metadata.json", split="train")
    dataset_f = "fingerspell"
    model = VideoPredictionModel(vocab_size, num_categories)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()

    # Predict videos for the sentence and ensure the order
    predicted_videos = predict_video_for_words(isl_sentence, word_to_idx, category_to_idx,model,dataset, dataset_f, output_folder)
    print("Predicted videos:", predicted_videos)
    # Concatenate videos in the correct order
    if predicted_videos:
        concatenated_video_path = os.path.join(output_folder, "concatenated_video.avi")
        concat_videos(predicted_videos, concatenated_video_path)

        # Convert concatenated video to MP4
        mp4_video_path = os.path.join(output_folder, "concatenated_video.mp4")
        convert_to_mp4(concatenated_video_path, mp4_video_path)
        output_csv = os.path.join(output_folder,'pose_data.csv')
        extract_pose(mp4_video_path,output_csv)
       
    else:
        print("No videos to concatenate.")

if __name__ == "__main__":
    main()
