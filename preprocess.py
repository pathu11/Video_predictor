import os
import json
import shutil
import random
from pathlib import Path

def split_dataset(dataset_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
    """
    Splits a dataset into training, validation, and test sets based on a given split ratio.
    
    Args:
        dataset_dir (str): Path to the input dataset directory containing categories and words.
        output_dir (str): Path to the output directory where the split datasets will be saved.
        split_ratio (tuple): A tuple containing the ratio of train, validation, and test splits.
                             Default is (0.7, 0.2, 0.1).
    """
    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    metadata = {}

    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        words = [w for w in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, w))]
        
        for word in words:
            word_path = os.path.join(category_path, word)
            videos = [f for f in os.listdir(word_path) if f.endswith(('.mp4', '.mov'))]
            random.shuffle(videos)

            train_size = int(len(videos) * split_ratio[0])
            val_size = int(len(videos) * split_ratio[1])

            splits = {
                'train': videos[:train_size],
                'val': videos[train_size:train_size + val_size],
                'test': videos[train_size + val_size:]
            }

            for split, split_videos in splits.items():
                split_dir = os.path.join(output_dir, split, category, word)
                os.makedirs(split_dir, exist_ok=True)
                for video in split_videos:
                    shutil.copy(os.path.join(word_path, video), os.path.join(split_dir, video))

            metadata[word] = {'category': category, 'videos': videos}

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

dataset_dir = "dataset"
output_dir = "processed_dataset"
split_dataset(dataset_dir, output_dir)
