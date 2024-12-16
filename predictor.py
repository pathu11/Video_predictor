import torch
import json
import os
import cv2
from tokenizer import preprocess_to_isl
from train import VideoPredictionModel
from train import VideoDataset

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


def predict_video_for_words(isl_sentence, word_to_idx, category_to_idx, model, dataset, output_folder):
    """
    Predict the video for each word in the ISL sentence.

    Args:
        isl_sentence (str): The sentence in ISL format.
        word_to_idx (dict): Mapping of words to indices.
        category_to_idx (dict): Mapping of categories to indices.
        model (VideoPredictionModel): The trained video prediction model.
        dataset (VideoDataset): Dataset containing video metadata.
        output_folder (str): Directory to save the predicted videos.

    Returns:
        list: List of tuples containing the category index and predicted video file name.
    """
    tokens = isl_sentence.split()
    token_indices = [word_to_idx[token] for token in tokens if token in word_to_idx]

    predicted_videos = []
    for token_idx in token_indices:
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

                    save_video(output_folder, f"predicted_video_{token_idx}.avi", frames)
                    predicted_videos.append((category_idx, f"predicted_video_{token_idx}.avi"))
                else:
                    print(f"Warning: No videos found for {category} - {word}")
            else:
                print(f"Warning: No category found for word: {word}")

    return predicted_videos


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


def main():
    """
    Main function to process the ISL sentence, predict videos, and save them.
    """
    input_text = input("Enter tokenized sentence: ")
    isl_sentence = preprocess_to_isl(input_text)
    print("Processed ISL Sentence:", isl_sentence)

    output_folder = "predicted_videos"
    clear_previous_videos(output_folder)

    with open("models/word_to_idx.json", "r") as f:
        word_to_idx = json.load(f)

    with open("models/category_to_idx.json", "r") as f:
        category_to_idx = json.load(f)

    model_path = "models/video_prediction_model.pth"
    vocab_size = len(word_to_idx)
    num_categories = len(category_to_idx)

    dataset = VideoDataset(metadata_file="processed_dataset/metadata.json", split="train")
    model = load_model(model_path, vocab_size, num_categories)

    predicted_videos = predict_video_for_words(isl_sentence, word_to_idx, category_to_idx, model, dataset, output_folder)
    print("Predicted Video Sequences:", predicted_videos)

    concat_videos([os.path.join(output_folder, filename) for filename in os.listdir(output_folder) if filename.endswith('.avi')],
                  os.path.join(output_folder, "concatenated_video.avi"))


if __name__ == "__main__":
    main()
