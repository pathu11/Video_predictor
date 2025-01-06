import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DistilBertModel
from torchvision import models, transforms
from torchvision.io import read_video
from transformers import AutoModel


class TextFeatureExtractor(nn.Module):
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.mean(dim=1)
    
class VideoFeatureExtractor(nn.Module):
    def __init__(self):
        super(VideoFeatureExtractor, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Remove classification head

    def forward(self, video):
        batch_features = []
        for batch_video in video: 
            video_features = []
            for frame in batch_video:  # Loop through frames in a single video
                frame = frame.unsqueeze(0)  # Add batch dimension for ResNet
                feature = self.resnet(frame)  # Extract features for a single frame
                video_features.append(feature.squeeze(0))  # Remove the batch dimension
            video_features = torch.stack(video_features)  # Stack features for all frames
            batch_features.append(video_features.mean(dim=0))  # Average across frames
        return torch.stack(batch_features)  # Stack all video features in the batch


class MultiModalModel(nn.Module):
    def __init__(self, text_model_name, video_model, num_classes):
        super(MultiModalModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.video_model = video_model
        self.fc = nn.Linear(768 + 2048, num_classes)  # Example sizes: 768 for text (BERT), 2048 for video features

    def forward(self, input_ids, attention_mask, video):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # Shape: [batch_size, 768]
        video_features = self.video_model(video)  # Shape: [batch_size, 2048]
        combined_features = torch.cat((text_features, video_features), dim=-1)  # Shape: [batch_size, 2816]
        return self.fc(combined_features)


# Define the dataset for video-text pairs
def safe_read_video(word_path):
    try:
        video, _, _ = read_video(word_path, pts_unit='sec')
        return video
    except Exception as e:
        print(f"Error reading video {word_path}: {e}")
        return None

class GestureDataset(Dataset):
    def __init__(self, dataset_dir, tokenizer, max_length=128):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        self.category_word_map = {}
        for category in self.categories:
            category_path = os.path.join(dataset_dir, category)
            words = [w for w in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, w))]
            self.category_word_map[category] = words
        # print(f"Categories and their words: {self.category_word_map}")
        self.samples = []
        for category, words in self.category_word_map.items():
            category_path = os.path.join(dataset_dir, category)
            for word in words:
                word_path = os.path.join(category_path, word)
                videos = [f for f in os.listdir(word_path) if f.endswith(('.mp4', '.mov'))]
                for video in videos:
                    self.samples.append((category, word, os.path.join(word_path, video)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        category, word, video_path = self.samples[idx]
        encoding = self.tokenizer(word, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        video, _, _ = read_video(video_path, pts_unit='sec')
        if video.size(0) == 0:
            print(f"Warning: Video at {video_path} is empty or could not be read.")
            raise ValueError(f"Invalid video file: {video_path}")
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        video_transformed = []
        for frame in video:
            if frame.size(-1) != 3:
                print(f"Skipping frame with unexpected channels: {frame.size(-1)}")
                continue
            frame = frame.permute(2, 0, 1)  # Convert to (C, H, W)
            frame = transforms.ToPILImage()(frame)
            video_transformed.append(transform(frame))
        if len(video_transformed) == 0:
            print(f"Warning: No valid frames found in video: {video_path}")
            raise ValueError(f"Video {video_path} has no valid frames.")
        target_length = 32
        if len(video_transformed) > target_length:
            video_transformed = video_transformed[:target_length]  
        elif len(video_transformed) < target_length:
            padding = target_length - len(video_transformed)
            video_transformed += [torch.zeros((3, 224, 224))] * padding 
        video = torch.stack(video_transformed)
        category_label = self.categories.index(category)
        word_label = self.category_word_map[category].index(word)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'video': video,
            'category_labels': torch.tensor(category_label),
            'word_labels': torch.tensor(word_label)
        }
def train_model(model, train_loader, val_loader, num_epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            video = batch['video']
            category_labels = batch['category_labels']

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, video=video)
            loss = criterion(output, category_labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_dataset = GestureDataset(dataset_dir='processed_datasetC/train', tokenizer=tokenizer)
val_dataset = GestureDataset(dataset_dir='processed_datasetC/val', tokenizer=tokenizer)
test_dataset = GestureDataset(dataset_dir='processed_datasetC/test', tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

text_model = "distilbert-base-uncased"
video_model = VideoFeatureExtractor()
# multi_modal_model = MultiModalModel(text_model, video_model, num_classes=16)
multi_modal_model = MultiModalModel("distilbert-base-uncased", video_model, num_classes=5)
train_model(multi_modal_model, train_loader, val_loader)
