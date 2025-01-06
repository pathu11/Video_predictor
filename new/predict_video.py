import pickle
import numpy as np
from preprocess_input import preprocess_input_text, preprocess_video_frame
from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    """Load the trained model and embeddings"""
    with open('./models/word_embeddings.pkl', 'rb') as f:
        word_embeddings = pickle.load(f)
    with open('./models/video_embeddings.pkl', 'rb') as f:
        video_embeddings = pickle.load(f)
    with open('./models/word_to_video_map.pkl', 'rb') as f:
        word_to_video_map = pickle.load(f)
    with open('./models/nearest_neighbors_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return word_embeddings, video_embeddings, word_to_video_map, model

def get_predicted_video(input_text, word_embeddings, model, word_to_video_map):
    """Predict the sign video corresponding to the input text"""
    input_embedding = preprocess_input_text(input_text)

    # Find the nearest video embedding for the input word
    distances, indices = model.kneighbors([input_embedding])
    predicted_video = word_to_video_map[list(word_to_video_map.keys())[indices[0][0]]]
    
    return predicted_video

if __name__ == "__main__":
    # Load model and embeddings
    word_embeddings, video_embeddings, word_to_video_map, model = load_model()

    # Take input from the user (word)
    input_word = input("Enter the word: ")
    
    # Predict and output the corresponding video
    predicted_video = get_predicted_video(input_word, word_embeddings, model, word_to_video_map)
    print(f"Predicted video for '{input_word}': {predicted_video}")
