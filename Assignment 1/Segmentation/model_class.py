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

from dataset_class import CamVidDataset, denormalize, class_dict, classes, label_to_class, class_to_label, label_to_color, color_to_label

class SegNet_Encoder(nn.Module):

    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet_Encoder, self).__init__()

        #SegNet Architecture
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers

        self.in_chn = in_chn
        self.out_chn = out_chn

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)
    def forward(self,x):
        #ENCODE LAYERS
        #Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
        x = F.relu(self.BNEn33(self.ConvEn33(x)))   
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
        x = F.relu(self.BNEn43(self.ConvEn43(x)))   
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
        x = F.relu(self.BNEn53(self.ConvEn53(x)))   
        x, ind5 = self.MaxEn(x)
        size5 = x.size()
        return x,[ind1,ind2,ind3,ind4,ind5],[size1,size2,size3,size4,size5]
    


class SegNet_Decoder(nn.Module):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet_Decoder, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        #implement the architecture.

         # stage 5:
        # Max Unpooling: Upsample using ind5 to size4
        # Channels: 512 → 512 → 512 (3 convolutions)
        # Batch Norm: Applied after each convolution
        # Activation: ReLU after each batch norm

        self.MaxDe = nn.MaxUnpool2d(2, stride=2)

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        
        #stage 4:
        # Max Unpooling: Upsample using ind4 to size3
        # Channels: 512 → 512 → 256 (3 convolutions)
        # Batch Norm: Applied after each convolution
        # Activation: ReLU after each batch norm

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)
                
        # Stage 3:
        # Max Unpooling: Upsample using ind3 to size2
        # Channels: 256 → 256 → 128 (3 convolutions)
        # Batch Norm: Applied after each convolution
        # Activation: ReLU after each batch norm

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)
        
        # Stage 2:
        # Max Unpooling: Upsample using ind2 to size1
        # Channels:  128 → 64 (2 convolutions)
        # Batch Norm: Applied after each convolution
        # Activation: ReLU after each batch norm

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)
        
        # Stage 1:
        # Max Unpooling: Upsample using ind1
        # Channels: 64 → out_chn (2 convolutions)
        # Batch Norm: Applied after each convolution
        # Activation: ReLU after the first convolution, no activation after the last one 

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(out_chn, momentum=BN_momentum)


        #For convolution use kernel size = 3, padding =1 
        #for max unpooling use kernel size=2 ,stride=2 
    def forward(self,x,indexes,sizes):
        ind1,ind2,ind3,ind4,ind5=indexes[0],indexes[1],indexes[2],indexes[3],indexes[4]
        size1,size2,size3,size4,size5=sizes[0],sizes[1],sizes[2],sizes[3],sizes[4]
        
        #DECODE LAYERS
        #Stage 5
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))
        
        #Stage 4
        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        #Stage 3
        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        #Stage 2
        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        #Stage 1
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.BNDe11(self.ConvDe11(x))
        return x
    

class SegNet_Pretrained(nn.Module):
    def __init__(self,encoder_weight_pth,in_chn=3, out_chn=32):
        super(SegNet_Pretrained, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.encoder=SegNet_Encoder(in_chn=self.in_chn,out_chn=self.out_chn)
        self.decoder=SegNet_Decoder(in_chn=self.in_chn,out_chn=self.out_chn)
        encoder_state_dict = torch.load(encoder_weight_pth,weights_only=True)

        # Load weights into the encoder
        self.encoder.load_state_dict(encoder_state_dict)

        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self,x):
        x,indexes,sizes=self.encoder(x)
        x=self.decoder(x,indexes,sizes)
        return x


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=32):
        super(DeepLabV3, self).__init__()

        self.model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
       
    def forward(self, x):
        return self.model(x)['out']
    
def train(model, train_loader, num_epochs = 10, lr = 0.001):

    wandb.watch(model, log="all", log_freq=32)

    # logging the hyperparameters
    # wandb.config = {
    #     "epochs": num_epochs,
    #     "lr": lr,
    #     "batch_size": 32,
    #     "optimizer": "Adam",
    #     "loss": "CrossEntropyLoss",
    #     "model": "DeepLabV3", # SegNet 
    #     "dataset": "CamVid Dataset"
    # }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr) # SegNet
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # DeepLabV3
    criterion = nn.CrossEntropyLoss()

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_train = 0

        # Training loop
        for inputs, labels, targets in train_loader:
            inputs, labels, targets = inputs.to(device), labels.to(device), targets.to(device)

            # print(inputs.shape)
            optimizer.zero_grad()
            model_outputs = model(inputs)
            loss = criterion(model_outputs, targets) 
            loss.backward()
            optimizer.step()


            train_loss += loss.item() * inputs.size(0)  # Accumulate the loss
            total_train += labels.size(0)

        # bookkeeping of losses
        avg_train_loss = train_loss / total_train
        train_losses.append(avg_train_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] \n")
        print(f"Training Loss: {avg_train_loss:.4f}")

        # logging the metrics to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": avg_train_loss,
        })

    # plot the training loss across epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.savefig("training_loss_plot.png")
    plt.show()

    wandb.log({"training_loss_plot": wandb.Image("training_loss_plot.png")})

    # Save the model weights
    # torch.save(model.decoder.state_dict(), "decoder.pth")
    torch.save(model.state_dict(), "deeplabv3.pth")

    # Save the model to wandb
    # wandb.save("decoder.pth")
    wandb.save("deeplabv3.pth")

    return model

if __name__ == '__main__':
    
    # initializing wandb
    wandb.login()
    wandb.init(project="CAMVid Image Segmentation", name="DeepLabV3") # SegNet Decoder
    
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
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # print(len(train_loader), len(test_loader))

    # creating dataloaders for deep lab v3
    train_loader = DataLoader(train_dataset, batch_size=9, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=9, shuffle=False)
    # print(len(train_loader), len(test_loader))

    # initializing the model
    # model = SegNet_Pretrained(encoder_weight_pth='encoder_model.pth', in_chn=3, out_chn=32)
    model = DeepLabV3(num_classes=32)

    # training the model
    model = train(model, train_loader, num_epochs=20, lr=0.001)

    # evaluating the model
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_loss = 0.0
    total_test = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    correct_pixels_per_class = {k: 0 for k in range(len(classes))} # true positives
    false_positive_per_class = {k: 0 for k in range(len(classes))} # false positives
    false_negative_per_class = {k: 0 for k in range(len(classes))} # false negatives
    intersection_per_class = {k: 0 for k in range(len(classes))}
    union_per_class = {k: 0 for k in range(len(classes))}
    total_ground_truth_pixels_per_class = {k: 0 for k in range(len(classes))}
    total_predicted_pixels_per_class = {k: 0 for k in range(len(classes))}

    with torch.no_grad():
        for inputs, labels, targets in test_loader:
            inputs, labels, targets = inputs.to(device), labels.to(device), targets.to(device)
            model_outputs = model(inputs)
            loss = criterion(model_outputs, targets)

            # model_outputs is of shape (batch_size, num_classes, height, width) i.e. raw logits
            # targets is of shape (batch_size, height, width) i.e. the ground truth labels

            ground_truth = targets.cpu().numpy()
            predictions = model_outputs.argmax(1).cpu().numpy()

            for c in range(len(classes)):
                correct_pixels_per_class[c] += np.logical_and(predictions == c, ground_truth == c).sum()
                intersection_per_class[c] += np.logical_and(predictions == c, ground_truth == c).sum()
                union_per_class[c] += np.logical_or(predictions == c, ground_truth == c).sum()
                total_ground_truth_pixels_per_class[c] += (ground_truth == c).sum()
                total_predicted_pixels_per_class[c] += (predictions == c).sum()
                false_positive_per_class[c] += np.logical_and(predictions == c, ground_truth != c).sum()
                false_negative_per_class[c] += np.logical_and(predictions != c, ground_truth == c).sum()

            test_loss += loss.item() * inputs.size(0)
            total_test += labels.size(0)

    avg_test_loss = test_loss / total_test
    print(f"Test Loss: {avg_test_loss:.4f}")

    pixelwise_accuracy_per_class = {}
    iou_per_class = {}
    dice_coeff_per_class = {}
    precision_per_class = {}
    recall_per_class = {}

    for c in range(len(classes)):
        if total_ground_truth_pixels_per_class[c] > 0:
            pixelwise_accuracy_per_class[c] = correct_pixels_per_class[c] / total_ground_truth_pixels_per_class[c]
        else:
            pixelwise_accuracy_per_class[c] = None

        if union_per_class[c] > 0:
            iou_per_class[c] = intersection_per_class[c] / union_per_class[c]
        else:
            iou_per_class[c] = None

        if total_ground_truth_pixels_per_class[c] + total_predicted_pixels_per_class[c] > 0:
            dice_coeff_per_class[c] = 2 * intersection_per_class[c] / (total_ground_truth_pixels_per_class[c] + total_predicted_pixels_per_class[c])
        else:
            dice_coeff_per_class[c] = None

        if correct_pixels_per_class[c] + false_positive_per_class[c] > 0:
            precision_per_class[c] = correct_pixels_per_class[c] / (correct_pixels_per_class[c] + false_positive_per_class[c])
        else:
            precision_per_class[c] = None

        if correct_pixels_per_class[c] + false_negative_per_class[c] > 0:
            recall_per_class[c] = correct_pixels_per_class[c] / (correct_pixels_per_class[c] + false_negative_per_class[c])
        else:
            recall_per_class[c] = None

    print("Pixelwise Accuracy per Class:")
    for c, acc in pixelwise_accuracy_per_class.items():
        print(f"Class {label_to_class[c]}: {acc:.4f}" if acc is not None else f"Class {label_to_class[c]}: No pixels in dataset")

    print("\nIoU per Class:")
    for c, iou in iou_per_class.items():
        print(f"Class {label_to_class[c]}: {iou:.4f}" if iou is not None else f"Class {label_to_class[c]}: No pixels in dataset")

    print("\nDice Coefficient per Class:")
    for c, dice in dice_coeff_per_class.items():
        print(f"Class {label_to_class[c]}: {dice:.4f}" if dice is not None else f"Class {label_to_class[c]}: No pixels in dataset")

    mIoU = np.mean([iou for iou in iou_per_class.values() if iou is not None])
    print("\nmIoU:", mIoU)

    print("\nPrecision per Class:")
    for c, precision in precision_per_class.items():
        print(f"Class {label_to_class[c]}: {precision:.4f}" if precision is not None else f"Class {label_to_class[c]}: No pixels in dataset")

    print("\nRecall per Class:")
    for c, recall in recall_per_class.items():
        print(f"Class {label_to_class[c]}: {recall:.4f}" if recall is not None else f"Class {label_to_class[c]}: No pixels in dataset")

    # binning the IoU values in intervals of 0.1
    iou_bins = np.arange(0, 1.1, 0.1)
    iou_bin_counts = np.zeros(len(iou_bins) - 1)

    for iou in iou_per_class.values():
        if iou is not None:
            iou_bin_counts[int(iou * 10)] += 1

    print("\nIoU Distribution:")
    for i in range(len(iou_bins) - 1):
        print(f"IoU Bin {iou_bins[i]}-{iou_bins[i + 1]}: {iou_bin_counts[i]} classes")

    
    # calculating the average precision and recall for all classes in each IoU bin
    iou_per_class = {k: v for k, v in iou_per_class.items() if v is not None}
    precision_per_iou_bin = {}
    recall_per_iou_bin = {}

    print("\nClasses per IoU Bin:")
    for i in range(len(iou_bins) - 1):
        lower = iou_bins[i]
        upper = iou_bins[i + 1]
        classes_in_bin = [c for c, iou in iou_per_class.items() if iou is not None and lower <= iou < upper]
        classes_names_in_bin = [label_to_class[c] for c in classes_in_bin]
        print(f"IoU Bin {lower}-{upper}: {classes_names_in_bin}")

        if len(classes_in_bin) > 0:
            precisions = [precision_per_class[c] for c in
                          classes_in_bin if precision_per_class[c] is not None]
            recalls = [recall_per_class[c] for c in
                       classes_in_bin if recall_per_class[c] is not None] 
    
            precision_per_iou_bin[(lower, upper)] = np.mean(precisions)
            recall_per_iou_bin[(lower, upper)] = np.mean(recalls)

    print("\nPrecision per IoU Bin:")
    for iou_bin, precision in precision_per_iou_bin.items():
        print(f"IoU Bin {iou_bin}: {precision:.4f}")

    print("\nRecall per IoU Bin:")
    for iou_bin, recall in recall_per_iou_bin.items():
        print(f"IoU Bin {iou_bin}: {recall:.4f}")

    # plotting the precision and recall per IoU bin using histogram
    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    bins = np.arange(0, 1, 0.1)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    precision_values = [precision_per_iou_bin.get((bins[i], bins[i+1]), 0) for i in range(len(bins)-1)]
    recall_values = [recall_per_iou_bin.get((bins[i], bins[i+1]), 0) for i in range(len(bins)-1)]
    
    r1 = np.arange(len(precision_values))
    r2 = [x + bar_width for x in r1]
    
    plt.bar(r1, precision_values, color='b', width=bar_width, edgecolor='grey', label='Precision')
    plt.bar(r2, recall_values, color='r', width=bar_width, edgecolor='grey', label='Recall')
    
    plt.xlabel('IoU Bins', fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(len(precision_values))], bin_labels)
    plt.ylabel('Metrics')
    plt.title('Precision and Recall per IoU Bin')
    plt.legend()
    plt.tight_layout()
    plt.savefig("precision_recall_iou_bin_plot.png")
    plt.show()

    wandb.log({"precision_recall_iou_bin_plot": wandb.Image("precision_recall_iou_bin_plot.png")})

    # logging the metrics to wandb
    wandb.log({"Test Loss": avg_test_loss})

    for c, acc in pixelwise_accuracy_per_class.items():
        wandb.log({f"Class {label_to_class[c]} Pixelwise Accuracy": acc})

    for c, iou in iou_per_class.items():
        wandb.log({f"Class {label_to_class[c]} IoU": iou})

    for c, dice in dice_coeff_per_class.items():
        wandb.log({f"Class {label_to_class[c]} Dice Coefficient": dice})

    wandb.log({"mIoU": mIoU})

    for c, precision in precision_per_class.items():
        wandb.log({f"Class {label_to_class[c]} Precision": precision})

    for c, recall in recall_per_class.items():
        wandb.log({f"Class {label_to_class[c]} Recall": recall})

    wandb.log({"IoU Distribution": iou_bin_counts})

    for iou_bin, precision in precision_per_iou_bin.items():
        wandb.log({f"IoU Bin {iou_bin} Precision": precision})

    for iou_bin, recall in recall_per_iou_bin.items():
        wandb.log({f"IoU Bin {iou_bin} Recall": recall})


    # selecting 3 classes which have IoU < 0.5
    classes_to_visualize = [c for c, iou in iou_per_class.items() if iou is not None and iou < 0.5][:3]

    images_of_class = {c: [] for c in classes_to_visualize}

    for image, label, target in test_dataset:
        image = image.unsqueeze(0).to(device)  # adding batch dimension
        prediction = model(image).argmax(1).squeeze(0).cpu()  # prediction and remove batch dimension
        image = image.squeeze(0).cpu()  # removing batch dimension
        for k in classes_to_visualize:
            if torch.sum(target == k).item() > 0:
                if len(images_of_class[k]) < 3:
                    images_of_class[k].append([image, label, target, prediction])

    
    for class_ in classes_to_visualize:
        for i, (image, label, target, prediction) in enumerate(images_of_class[class_]):
            image_denorm = denormalize(image, mean, std).clamp(0, 1)
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(image_denorm.permute(1, 2, 0))
            ax[0].set_title("Image")
            ax[0].axis("off")

            class_label = class_
            color = [c / 255.0 for c in label_to_color[class_label]]

            # Ground Truth Mask
            label_reshaped = label.permute(1, 2, 0)
            mask_label = torch.zeros_like(label_reshaped)
            indices = torch.where(target == class_label)
            mask_label[indices[0], indices[1], :] = torch.tensor(color, dtype=mask_label.dtype)
            ax[1].imshow(mask_label)
            ax[1].set_title("Ground Truth mask")
            ax[1].axis("off")

            # Prediction Mask
            mask_prediction = torch.zeros((prediction.shape[0], prediction.shape[1], 3))
            indices = torch.where(prediction == class_label)
            mask_prediction[indices[0], indices[1], :] = torch.tensor(color, dtype=mask_prediction.dtype)
            ax[2].imshow(mask_prediction)
            ax[2].set_title("Prediction mask")
            ax[2].axis("off")

            plt.suptitle(f"Class {label_to_class[class_]} Image {i + 1}")
            plt.savefig(f"class_{label_to_class[class_]}_image_{i + 1}.png")
            plt.show()
            wandb.log({f"Class {label_to_class[class_]} Image {i + 1}": wandb.Image(f"class_{label_to_class[class_]}_image_{i + 1}.png")})


    wandb.finish()

    