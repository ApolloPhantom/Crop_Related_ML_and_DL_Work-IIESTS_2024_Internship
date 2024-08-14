#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
print(torch.__version__)
print(torch.cuda.is_available())


# In[2]:


from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.io import read_image

torch.device("cuda")


# In[3]:


from torchvision import transforms
h,w = 64,64
data_transform = transforms.Compose([
    transforms.Resize(size=(h,w)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


# In[4]:


import random
from PIL import Image
from pathlib import Path


random.seed(42)

data_path = Path("../Data/Pest_Prediction_Pro/")
image_path = data_path / "Pest Prediction"
train_dir = image_path / "train"
test_dir = image_path / "test"
image_path_list = list(image_path.glob("*/*/*.jpg"))
num_subfolders = len([f for f in train_dir.iterdir() if f.is_dir()])

print(num_subfolders)
def plot_transformed_images(image_paths, transform, n=5, seed=11):
    seed = random.randint(1,100)
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list, 
                        transform=data_transform, 
                        n=3)


# In[5]:


train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)


# In[6]:


batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=10,pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=10,pin_memory=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# In[7]:


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from matplotlib.animation import FuncAnimation
import time
import torch.optim as optim

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=5, num_classes=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = torch.compile(model)  # Compile the model for faster training
    model.to(device)
    model.train()
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    epoch_times = []

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)

        end_time = time.time()
        epoch_times.append(end_time - start_time)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%, Time: {end_time - start_time:.2f}s")
        
        
        model.train()

    # Average time per epoch
    avg_epoch_time = np.mean(epoch_times)
    print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, num_epochs + 1), y=train_losses, marker='o', label='Train Loss')
    sns.lineplot(x=range(1, num_epochs + 1), y=test_losses, marker='o', label='Test Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"../Results/Loss_{model.__class__.__name__}.png")
    plt.show()
    
    # Plot training and test accuracy over epochs
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, num_epochs + 1), y=train_accuracies, marker='o', label='Train Accuracy')
    sns.lineplot(x=range(1, num_epochs + 1), y=test_accuracies, marker='o', label='Test Accuracy')
    plt.title('Training and Test Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f"../Results/Accu_{model.__class__.__name__}.png")
    plt.show()
    

    # Confusion matrix plot
    conf_mat = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"../Results/ConfusionMatrix_{model.__class__.__name__}.png")
    plt.show()
    
    # ROC curve and AUC for each class
    all_labels = label_binarize(all_labels, classes=[i for i in range(num_classes)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], np.array(all_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Each Class')
    plt.legend(loc="center right", bbox_to_anchor=(1, 0.5))
    plt.savefig(f"../Results/ROC_AUC_{model.__class__.__name__}.png")
    plt.show()
    
    # Radar chart for accuracy of each class
    class_accuracies = [accuracy_score(np.array(all_labels)[:, i], np.array(all_preds) == i) for i in range(num_classes)]
    categories = [f'Class {i}' for i in range(num_classes)]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    class_accuracies += class_accuracies[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.fill(angles, class_accuracies, color='blue', alpha=0.25)
    ax.plot(angles, class_accuracies, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Class-wise Accuracy')
    plt.savefig(f"../Results/Radar{model.__class__.__name__}.png")
    plt.show()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
num_epochs = 20
learn_rate = 0.001


# In[8]:


import torchvision.models as models


# In[9]:


model1 = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
model1.classifier[6] = torch.nn.Linear(model1.classifier[6].in_features, 12)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=learn_rate)
torch.cuda.empty_cache()
train_model(model1,train_dataloader,test_dataloader,criterion, optimizer, num_epochs=num_epochs)
torch.cuda.empty_cache()


# In[10]:


model1 = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
model1.classifier[6] = torch.nn.Linear(model1.classifier[6].in_features, 12)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=learn_rate)
torch.cuda.empty_cache()
train_model(model1,train_dataloader,test_dataloader,criterion, optimizer, num_epochs=num_epochs)
torch.cuda.empty_cache()


# In[11]:


model1 = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
model1.classifier[6] = torch.nn.Linear(model1.classifier[6].in_features, 12)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=learn_rate)
torch.cuda.empty_cache()
train_model(model1,train_dataloader,test_dataloader,criterion, optimizer, num_epochs=num_epochs)
torch.cuda.empty_cache()


# In[12]:


model1 = models.vgg13_bn(weights=models.VGG13_BN_Weights.DEFAULT)
model1.classifier[6] = torch.nn.Linear(model1.classifier[6].in_features, 12)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=learn_rate)
torch.cuda.empty_cache()
train_model(model1,train_dataloader,test_dataloader,criterion, optimizer, num_epochs=num_epochs)
torch.cuda.empty_cache()


# In[13]:


model1 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model1.classifier[6] = torch.nn.Linear(model1.classifier[6].in_features, 12)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=learn_rate)
torch.cuda.empty_cache()
train_model(model1,train_dataloader,test_dataloader,criterion, optimizer, num_epochs=num_epochs)
torch.cuda.empty_cache()


# In[14]:


model1 = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
model1.classifier[6] = torch.nn.Linear(model1.classifier[6].in_features, 12)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=learn_rate)
torch.cuda.empty_cache()
train_model(model1,train_dataloader,test_dataloader,criterion, optimizer, num_epochs=num_epochs)
torch.cuda.empty_cache()


# In[9]:


model1 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
model1.classifier[6] = torch.nn.Linear(model1.classifier[6].in_features, 12)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=learn_rate)
torch.cuda.empty_cache()
train_model(model1,train_dataloader,test_dataloader,criterion, optimizer, num_epochs=num_epochs)
torch.cuda.empty_cache()


# In[10]:


model1 = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
model1.classifier[6] = torch.nn.Linear(model1.classifier[6].in_features, 12)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=learn_rate)
torch.cuda.empty_cache()
train_model(model1,train_dataloader,test_dataloader,criterion, optimizer, num_epochs=num_epochs)
torch.cuda.empty_cache()

