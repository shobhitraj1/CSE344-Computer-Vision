import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
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

dir = "../data/CamVid/"

class_dict = pd.read_csv(dir + "class_dict.csv")
class_dict = class_dict.set_index('name').T.to_dict('list')
# print(class_dict)

classes = list(class_dict.keys())
label_to_class = {i: classes[i] for i in range(len(classes))}
class_to_label = {classes[i]: i for i in range(len(classes))}
label_to_color = {i: class_dict[label_to_class[i]] for i in range(len(classes))}
color_to_label = {tuple(class_dict[label_to_class[i]]): i for i in range(len(classes))}


class CamVidDataset(Dataset):
    """
    Custom Dataset for CamVid Image Segmentation
    """
    def __init__(self, img_dir, labels_dir, transform=None, label_transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.label_transform = label_transform
        self.img_names = os.listdir(img_dir)
        self.images = []
        self.labels = []
        self.targets = []

        for img_name in self.img_names:
            img_path = os.path.join(self.img_dir, img_name)
            label_path = os.path.join(self.labels_dir, img_name.replace(".png", "_L.png"))
            image = Image.open(img_path)
            label = Image.open(label_path)
            if self.transform:
                image = self.transform(image)
            if self.label_transform:
                label = self.label_transform(label)

            self.images.append(image)
            self.labels.append(label)

            label_reshaped = label.permute(1, 2, 0)
            color_reshaped = (label_reshaped * 255).byte()
            h, w, _ = color_reshaped.shape
            color_flat = color_reshaped.view(-1, 3).tolist()
            mapped = [color_to_label.get(tuple(c), 30) for c in color_flat]
            target = torch.tensor(mapped, dtype=torch.int64).view(h, w)

            self.targets.append(target)          

        # print(len(self.images), len(self.labels), len(self.targets))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.targets[idx]


def denormalize(image, mean, std):
    """
    Denormalize the image for visualization purposes
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = image * std + mean
    return image


if __name__ == "__main__":

    # initializing wandb
    wandb.login()
    wandb.init(project="CAMVid Image Segmentation", name="Data Visualization")
    
    train_images = dir + "train"
    train_labels = dir + "train_labels"
    test_images = dir + "test_images"
    test_labels = dir + "test_labels"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((360, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    label_transform = transforms.Compose([
        transforms.Resize((360, 480), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    # loading the dataset
    train_dataset = CamVidDataset(train_images, train_labels, transform=transform, label_transform=label_transform)
    test_dataset = CamVidDataset(test_images, test_labels, transform=transform, label_transform=label_transform)
    # print(len(train_dataset), len(test_dataset))
    
    # creating dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # print(len(train_loader), len(test_loader))


    sample, label, target = train_dataset[0]
    # print(sample.shape, label.shape, target.shape)

    # # denormalize the image for visualization purposes
    # sample = denormalize(sample, mean, std).clamp(0, 1)
    # plt.imshow(sample.permute(1, 2, 0))
    # plt.show()

    images_of_class = {k: [] for k in range(len(classes))}

    # plotting class distribution using bar plot
    class_count = {k: 0 for k in range(len(classes))}
    image_count_per_class = {label_to_class[k]: 0 for k in range(len(classes))}
    for image, label, target in train_dataset:
        for k in range(len(classes)):
            class_count[k] += torch.sum(target == k).item()
            if torch.sum(target == k).item() > 0:
                image_count_per_class[label_to_class[k]] += 1
                if len(images_of_class[k]) < 2:
                    images_of_class[k].append([image, label, target])

    class_freq = {label_to_class[k]: v for k, v in class_count.items()}
    print(f"Total pixels = {sum(class_freq.values())}")
    print(f"Total pixels = {360 * 480 * len(train_dataset)}")
    plt.figure(figsize=(12, 6))
    plt.bar(class_freq.keys(), class_freq.values())
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Train Dataset Class Distribution wrt Pixels")
    plt.savefig("train_pixel_distribution.png")
    plt.show()
    wandb.log({"Train Dataset Class Distribution wrt Pixels": wandb.Image("train_pixel_distribution.png")})

    plt.figure(figsize=(12, 6))
    plt.bar(image_count_per_class.keys(), image_count_per_class.values())
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Train Dataset Class Distribution wrt Images")
    plt.savefig("train_image_distribution.png")
    plt.show()
    wandb.log({"Train Dataset Class Distribution wrt Images": wandb.Image("train_image_distribution.png")})
    # labels where the class frequency is max/min
    print([(k,v) for k, v in class_freq.items() if v == max(class_freq.values())])
    print([(k,v) for k, v in class_freq.items() if v == min(class_freq.values())])
    # labels where the image count is max/min
    print([(k,v) for k, v in image_count_per_class.items() if v == max(image_count_per_class.values())])
    print([(k,v) for k, v in image_count_per_class.items() if v == min(image_count_per_class.values())])
    
    class_count = {k: 0 for k in range(len(classes))}
    image_count_per_class = {label_to_class[k]: 0 for k in range(len(classes))}
    for _, _, target in test_dataset:
        for k in range(len(classes)):
            class_count[k] += torch.sum(target == k).item()
            if torch.sum(target == k).item() > 0:
                image_count_per_class[label_to_class[k]] += 1

    class_freq = {label_to_class[k]: v for k, v in class_count.items()}
    print(f"Total pixels = {sum(class_freq.values())}")
    print(f"Total pixels = {360 * 480 * len(test_dataset)}")
    plt.figure(figsize=(12, 6))
    plt.bar(class_freq.keys(), class_freq.values())
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Test Dataset Class Distribution wrt Pixels")
    plt.savefig("test_pixel_distribution.png")
    plt.show()
    wandb.log({"Test Dataset Class Distribution wrt Pixels": wandb.Image("test_pixel_distribution.png")})

    plt.figure(figsize=(12, 6))
    plt.bar(image_count_per_class.keys(), image_count_per_class.values())
    plt.xticks(rotation=90)
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Test Dataset Class Distribution wrt Images")
    plt.savefig("test_image_distribution.png")
    plt.show()
    wandb.log({"Test Dataset Class Distribution wrt Images": wandb.Image("test_image_distribution.png")})
    # labels where the class frequency is max/min
    print([(k,v) for k, v in class_freq.items() if v == max(class_freq.values())])
    print([(k,v) for k, v in class_freq.items() if v == min(class_freq.values())])
    # labels where the image count is max/min
    print([(k,v) for k, v in image_count_per_class.items() if v == max(image_count_per_class.values())])
    print([(k,v) for k, v in image_count_per_class.items() if v == min(image_count_per_class.values())])

    figsize = (12,3)
    for class_name in images_of_class:
        fig, axs = plt.subplots(1, 4, figsize=figsize)
        fig.suptitle(f"Class: {class_name}", fontsize=16)

        for j in range(len(images_of_class[class_name])):
            image, label, target = images_of_class[class_name][j]
            class_label = class_to_label[class_name]
            color = [c / 255.0 for c in label_to_color[class_label]]
            label_reshaped = label.permute(1, 2, 0)
            mask_label = torch.zeros_like(label_reshaped)
            indices = torch.where(target == class_label)
            mask_label[indices[0], indices[1], :] = torch.tensor(color, dtype=mask_label.dtype)
            sample_denorm = denormalize(image, mean, std).clamp(0, 1)
            
            axs[j*2].imshow(sample_denorm.permute(1, 2, 0))
            axs[j*2].set_title(f"Sample Image {j+1}")
            axs[j*2].axis("off")

            axs[j*2 + 1].imshow(mask_label)
            axs[j*2 + 1].set_title(f"Masked Label {j+1}")
            axs[j*2 + 1].axis("off")

        plt.tight_layout()
        plt.savefig(f"{class_name}_mask_visualization.png")
        plt.show()


    wandb.finish()

