import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.manifold import TSNE
import os
import sys
import torch    
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image
import wandb

np.random.seed(2022482)
torch.manual_seed(2022482)

class_labels = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear': 4, 'dog': 5, 'roe_deer': 6, 'sika_deer': 7, 'wild_boar': 8, 'people': 9}
# print(class_labels)

class RussianWildlifeDataset(Dataset):
    """
    Custom Dataset for Russian Wildlife Image Classification
    """
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.labels = []
        for folder in os.listdir(img_dir):
            folder_path = os.path.join(img_dir, folder)
            for file in os.listdir(folder_path):
                img = Image.open(os.path.join(folder_path, file))
                label = class_labels[folder]
                if self.transform:
                    img = self.transform(img)
                if self.target_transform:
                    label = self.target_transform(label)
                self.data.append(img)
                self.labels.append(label)
            # print(f"{folder}: {len(os.listdir(folder_path))}")
            # print(list(zip(self.data[-5:], self.labels[-5:])))
        
        # print(len(self.data))
        # print(len(self.labels))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

if __name__ == "__main__":

    # initializing wandb
    wandb.login()
    wandb.init(project="Russian Wildlife Image Classification")
    
    img_dir = "HW1-Resources/data/russian-wildlife-dataset/Cropped_final"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # loading the dataset
    dataset = RussianWildlifeDataset(img_dir, transform=transform)
    # print(len(dataset))

    # train-val split
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=2022482, stratify=dataset.labels)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # print(len(train_dataset))
    # print(len(val_dataset))

    # creating dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    all_labels = np.array(dataset.labels)
    # print(np.bincount(all_labels, minlength=10)) 

    train_counts = np.bincount(all_labels[train_indices], minlength=10)
    # print(train_counts) 

    val_counts = np.bincount(all_labels[val_indices], minlength=10)
    # print(val_counts) 

    # plotting class distribution using bar plot
    fig, ax = plt.subplots(3, 1, figsize=(12, 15))
    ax[0].bar(class_labels.keys(), np.bincount(all_labels, minlength=10))
    ax[0].set_title("Distribution of classes in Entire Dataset")
    ax[0].set_xlabel("Class")
    ax[0].set_ylabel("Count")
    ax[1].bar(class_labels.keys(), train_counts)
    ax[1].set_title("Distribution of classes in Training Set")
    ax[1].set_xlabel("Class")
    ax[1].set_ylabel("Count")
    ax[2].bar(class_labels.keys(), val_counts)
    ax[2].set_title("Distribution of classes in Validation Set")
    ax[2].set_xlabel("Class")
    ax[2].set_ylabel("Count")

    plt.subplots_adjust(hspace=0.5)
    fig.savefig("class_distribution.png", dpi=300)
    plt.show()

    # plotting class distribution using pie chart
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].pie(np.bincount(all_labels, minlength=10), labels=class_labels.keys(), autopct='%1.1f%%')
    ax[0].set_title("Entire Dataset")
    ax[1].pie(train_counts, labels=class_labels.keys(), autopct='%1.1f%%')
    ax[1].set_title("Training Set")
    ax[2].pie(val_counts, labels=class_labels.keys(), autopct='%1.1f%%')
    ax[2].set_title("Validation Set")
    
    fig.savefig("class_distribution_pie.png", dpi=300)
    plt.show()

    wandb.log({"class_distribution": wandb.Image("class_distribution.png")})
    wandb.log({"class_distribution_pie": wandb.Image("class_distribution_pie.png")})
