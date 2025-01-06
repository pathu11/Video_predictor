import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import json

class VideoDataset(Dataset):
    """
    A custom dataset class for loading video data based on metadata.
    Args:
        metadata_file (str): Path to the JSON file containing metadata for the dataset.
        split (str): The split of the dataset (e.g., 'train', 'val', 'test').
    """
    def __init__(self, metadata_file, split):
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.split = split
        self.data = []

        for word, info in self.metadata.items():
            videos = info['videos']
            category = info['category']
            for video in videos:
                self.data.append((word, category, video))
        
        self.word_to_idx = {word: idx for idx, word in enumerate(set(item[0] for item in self.data))}
        self.category_to_idx = {category: idx for idx, category in enumerate(set(item[1] for item in self.data))}
        self.idx_to_video = {idx: video for idx, (_, _, video) in enumerate(self.data)}
    
    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a dataset item by its index.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            tuple: A tuple containing the word index, category index, and a dummy video tensor.
        """
        word, category, video = self.data[idx]
        
        word_idx = torch.tensor(self.word_to_idx[word], dtype=torch.long)
        category_idx = torch.tensor(self.category_to_idx[category], dtype=torch.long)
        video_tensor = torch.zeros((1, 128, 128)) 
        
        return word_idx, category_idx, video_tensor


class VideoPredictionModel(nn.Module):
    """
    A simple neural network model for video and category prediction.

    Args:
        vocab_size (int): Number of unique words in the dataset.
        num_categories (int): Number of unique categories in the dataset.
    """
    def __init__(self, vocab_size, num_categories):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)  
        self.hidden = nn.Linear(128, 64)              
        self.fc = nn.Linear(64, num_categories)        
        self.video_fc = nn.Linear(64, 1)               
        
    def forward(self, x):
        """
        Defines the forward pass for the model.

        Args:
            x (Tensor): Input tensor of word indices.
        
        Returns:
            tuple: A tuple containing category prediction and video prediction score.
        """
        x = self.embedding(x)  
        x = x.view(x.size(0), -1)  
        x = self.hidden(x)         
        x = torch.relu(x)          
        category_out = self.fc(x)  
        video_out = self.video_fc(x)  
        
        return category_out, video_out


def train_model():
    """
    Trains the VideoPredictionModel using the VideoDataset.
    This function loads the dataset, initializes the model, and trains it for 10 epochs.
    After training, the model and vocabulary mappings are saved to disk.
    """
    dataset = VideoDataset(metadata_file="processed_dataset/metadata.json", split="train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    vocab_size = len(dataset.word_to_idx)
    num_categories = len(dataset.category_to_idx)
    model = VideoPredictionModel(vocab_size, num_categories)
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  

    for epoch in range(10):
        for words, categories, videos in dataloader:  
            category_preds, video_preds = model(words)  
            loss = criterion(category_preds, categories)  
            
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  

        print(f"Epoch {epoch}, Loss: {loss.item()}")  
    
    torch.save(model.state_dict(), "models/video_prediction_model.pth")
    
    with open("models/word_to_idx.json", "w") as f:
        json.dump(dataset.word_to_idx, f)
    with open("models/category_to_idx.json", "w") as f:
        json.dump(dataset.category_to_idx, f)

train_model()
