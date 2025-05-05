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

from dataset_class import RussianWildlifeDataset, class_labels

np.random.seed(2022482)
torch.manual_seed(2022482)

# custom ConvNet class
class ConvNet(nn.Module):
    """
    Custom Convolutional Neural Network for image classification
    """ 
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(128 * 14 * 14, 10) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# model = ConvNet()
# print(model)

class ResNet18(nn.Module):
    def __init__(self, num_classes: int=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def train(model, train_loader, val_loader, num_epochs = 10, lr = 0.001, model_name="ConvNet"):

    wandb.watch(model, log="all", log_freq=32)

    # logging the hyperparameters
    # wandb.config = {
    #     "epochs": num_epochs,
    #     "lr": lr,
    #     "batch_size": 32,
    #     "optimizer": "Adam",
    #     "loss": "CrossEntropyLoss",
    #     "model": "ResNet18", # "ConvNet"
    #     "dataset": "Russian Wildlife Dataset"
    # }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # print(inputs.shape)
            optimizer.zero_grad()
            model_outputs = model(inputs)
            loss = criterion(model_outputs, labels)
            loss.backward()
            optimizer.step()


            train_loss += loss.item() * inputs.size(0)  # Accumulate the loss
            _, predicted = torch.max(model_outputs, 1)  # Get the predicted class
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # bookkeeping of losses
        avg_train_loss = train_loss / total_train
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():  # No need to calculate gradients for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                model_outputs = model(inputs)
                loss = criterion(model_outputs, labels)

                val_loss += loss.item() * inputs.size(0)  # Accumulate the loss
                _, predicted = torch.max(model_outputs, 1)  # Get the predicted class
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / total_val
        val_losses.append(avg_val_loss)

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        f1 = f1_score(all_labels, all_predictions, average='weighted')
        f1_scores.append(f1)

        print(f"Epoch [{epoch + 1}/{num_epochs}] \n")
        print(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%, Validation F1 Score: {f1:.4f}")

        # logging the metrics to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "Training Accuracy": train_accuracy,
            "Validation Accuracy": val_accuracy,
            "Validation F1 Score": f1
        })

    # Plot the losses and accuracies accross epochs in 2 subplots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Loss per Epoch")

    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy per Epoch")

    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("loss_accuracy_plot.png")
    plt.show()

    wandb.log({"loss_accuracy_plot": wandb.Image("loss_accuracy_plot.png")})

    # plot the f1 score across epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), f1_scores, label="F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Epoch on Val Set")
    plt.legend()
    plt.savefig("f1_score_plot.png")
    plt.show()

    wandb.log({"f1_score_plot": wandb.Image("f1_score_plot.png")})

    # Save the model weights
    torch.save(model.state_dict(), f"{model_name}.pth")

    # Save the model to wandb
    wandb.save(f"{model_name}.pth")

    return model


augmentations = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])

def train_augment(model, train_loader, val_loader, num_epochs = 10, lr = 0.001, model_name="ConvNet"):

    wandb.watch(model, log="all", log_freq=32)

    # logging the hyperparameters
    # wandb.config = {
    #     "epochs": num_epochs,
    #     "lr": lr,
    #     "batch_size": 32,
    #     "optimizer": "Adam",
    #     "loss": "CrossEntropyLoss",
    #     "model": "ConvNet", # "ResNet18"
    #     "dataset": "Russian Wildlife Dataset"
    # }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            augmented_inputs = []
            augmented_labels = []
            for input_img, label in zip(inputs, labels):
                augmented_inputs.append(input_img)
                augmented_labels.append(label)
                for i in range(3):
                    augmented_image = augmentations(input_img)
                    augmented_inputs.append(augmented_image)
                    augmented_labels.append(label)
            augmented_inputs = torch.stack(augmented_inputs)
            augmented_labels = torch.tensor(augmented_labels).to(device)
                            
            # print(augmented_inputs.shape)
            # print(augmented_labels.shape)
            optimizer.zero_grad()
            model_outputs = model(augmented_inputs)
            loss = criterion(model_outputs, augmented_labels)
            loss.backward()
            optimizer.step()


            train_loss += loss.item() * augmented_inputs.size(0)  # Accumulate the loss
            _, predicted = torch.max(model_outputs, 1)  # Get the predicted class
            correct_train += (predicted == augmented_labels).sum().item()
            total_train += augmented_labels.size(0)

        # bookkeeping of losses
        avg_train_loss = train_loss / total_train
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():  # No need to calculate gradients for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                model_outputs = model(inputs)
                loss = criterion(model_outputs, labels)

                val_loss += loss.item() * inputs.size(0)  # Accumulate the loss
                _, predicted = torch.max(model_outputs, 1)  # Get the predicted class
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / total_val
        val_losses.append(avg_val_loss)

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        f1 = f1_score(all_labels, all_predictions, average='weighted')
        f1_scores.append(f1)

        print(f"Epoch [{epoch + 1}/{num_epochs}] \n")
        print(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%, Validation F1 Score: {f1:.4f}")

        # logging the metrics to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "Training Accuracy": train_accuracy,
            "Validation Accuracy": val_accuracy,
            "Validation F1 Score": f1
        })

    # Plot the losses and accuracies accross epochs in 2 subplots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Loss per Epoch")

    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy per Epoch")

    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("loss_accuracy_plot.png")
    plt.show()

    wandb.log({"loss_accuracy_plot": wandb.Image("loss_accuracy_plot.png")})

    # plot the f1 score across epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), f1_scores, label="F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score per Epoch on Val Set")
    plt.legend()
    plt.savefig("f1_score_plot.png")
    plt.show()

    wandb.log({"f1_score_plot": wandb.Image("f1_score_plot.png")})

    # Save the model weights
    torch.save(model.state_dict(), f"{model_name}.pth")

    # Save the model to wandb
    wandb.save(f"{model_name}.pth")

    return model


if __name__ == "__main__":

    # initializing wandb
    wandb.login()
    wandb.init(project="Russian Wildlife Image Classification", config={
        "epochs": 10,
        "lr": 0.001,
        "batch_size": 32,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "model": "ResNet18", # "ConvNet"
        "dataset": "Russian Wildlife Dataset"
    })

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

    # initializing the model
    # model = ConvNet()
    model = ResNet18(num_classes=10)

    # training the model
    model = train(model, train_loader, val_loader, num_epochs=10, lr=0.001, model_name="resnet") # "convnet" for ConvNet model
    # model = train_augment(model, train_loader, val_loader, num_epochs=10, lr=0.001, model_name="resnet_aug")

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_labels = []
    all_predictions = []

    misclassified_images = []
    for i in range(10):
        misclassified_images.append([])
     
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            model_outputs = model(inputs)
            _, predicted = torch.max(model_outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            for label in range(len(labels)):
                if labels[label] != predicted[label] and len(misclassified_images[labels[label].item()]) < 3:
                    misclassified_images[labels[label].item()].append((inputs[label], labels[label].item(), predicted[label].item()))
       
    # find the final f1 score
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"Final F1 Score: {f1:.4f}")

    # find the final accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Final Accuracy: {accuracy*100:.2f}%")

    # plot the final confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()

    wandb.log({"f1_score": f1})
    wandb.log({"accuracy": accuracy})
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

    labels_to_class = {class_labels[i]: i for i in class_labels}

    # visualizing the misclassified images
    fig, ax = plt.subplots(10, 3, figsize=(15, 30))
    for i in range(10):
        for j in range(3):
            if j < len(misclassified_images[i]): 
                img, label, prediction = misclassified_images[i][j]
                img = img.permute(1, 2, 0).cpu().numpy()
                ax[i, j].imshow(img)
                ax[i, j].set_title(f"True: {labels_to_class[label]}, Predicted: {labels_to_class[prediction]}")
            else:
                ax[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("misclassified_images.png")
    plt.show()

    wandb.log({"misclassified_images": wandb.Image("misclassified_images.png")})

    # removing the last layer
    model.model.fc = nn.Identity()
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # feature vectors of train set
    train_features = []
    train_labels = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = model(inputs)
            train_features.extend(features.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

    # feature vectors of val set
    val_features = []
    val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = model(inputs)
            val_features.extend(features.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())


    tsne = TSNE(n_components=2, random_state=2022482)
    train_tsne = tsne.fit_transform(np.array(train_features))
    val_tsne = tsne.fit_transform(np.array(val_features))

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=train_tsne[:, 0], y=train_tsne[:, 1], hue=train_labels, palette='tab10')
    plt.title("tSNE Plot for Train Set")
    plt.savefig("train_tsne.png")
    plt.show()

    wandb.log({"train_tsne": wandb.Image("train_tsne.png")})

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=val_tsne[:, 0], y=val_tsne[:, 1], hue=val_labels, palette='tab10')
    plt.title("tSNE Plot for Validation Set")
    plt.savefig("val_tsne.png")
    plt.show()

    wandb.log({"val_tsne": wandb.Image("val_tsne.png")})

    tsne = TSNE(n_components=3, random_state=2022482)
    val_tsne = tsne.fit_transform(np.array(val_features))

    fig = plt.figure(figsize=(10, 8))   
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(val_tsne[:, 0], val_tsne[:, 1], val_tsne[:, 2], c=val_labels, cmap='tab10')
    plt.title("tSNE Plot for Validation Set")
    plt.savefig("val_tsne_3d.png")
    plt.show()

    wandb.log({"val_tsne_3d": wandb.Image("val_tsne_3d.png")})

    labels_to_class = {class_labels[i]: i for i in class_labels}

    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    for i in range(5):
        images, labels = [], []
        sample_img, sample_label = dataset[i*2000+80]
        images.append(sample_img)
        labels.append(sample_label)
        for _ in range(3):
            augmented_img = augmentations(sample_img)
            images.append(augmented_img)
            labels.append(sample_label)

        for j in range(4):
            axes[i, j].imshow(images[j].permute(1, 2, 0))
            axes[i, j].set_title(labels_to_class[labels[j]])
            axes[i, j].axis("off")
    
    plt.tight_layout()
    plt.savefig("augmented_images.png")
    plt.show()

    wandb.log({"augmented_images": wandb.Image("augmented_images.png")})

    wandb.finish()

