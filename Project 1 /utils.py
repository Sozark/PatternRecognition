# utils.py
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def load_mnist_images(folder_path):
    """Loads MNIST images from folders named 0-9."""
    images, labels = [], []
    for label in range(10):
        digit_folder = os.path.join(folder_path, str(label))
        for filename in os.listdir(digit_folder):
            img_path = os.path.join(digit_folder, filename)
            img = Image.open(img_path).convert("L")
            img_array = np.array(img) / 255.0  # normalize [0,1]
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

def train_test_split(X, y, train_ratio=0.8):
    """Random 80/20 train/test split"""
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(train_ratio * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
