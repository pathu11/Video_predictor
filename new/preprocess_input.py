import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

# Initialize the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load pre-trained ResNet model for video frame feature extraction
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

def get_text_embedding(text):
    """Generate BERT embeddings for the input text"""
    inputs = tokenizer(text, return_tensors='pt')
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

def get_video_feature(video_frame_path):
    """Extract feature vector from a single video frame using ResNet"""
    image = Image.open(video_frame_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = resnet_model(input_tensor)
    return output.squeeze().detach().numpy()

def preprocess_input_text(input_text):
    """Preprocess input text for prediction"""
    return get_text_embedding(input_text)

def preprocess_video_frame(video_frame_path):
    """Preprocess a video frame"""
    return get_video_feature(video_frame_path)

