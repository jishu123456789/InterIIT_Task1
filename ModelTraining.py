# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 23:19:29 2024

@author: jishu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision.models as models

from DataDownload import Dog_Breed_Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256 , 256)) , 
    transforms.ToTensor()
    
    ])
images_dir = 'dog_breed'

train_dataset = Dog_Breed_Dataset(images_dir , transform)
test_dataset = Dog_Breed_Dataset(images_dir , transform , train = False)

num_classes = len(set(train_dataset.labels))

# Define a more complex classifier head
class CustomClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomClassifierHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)  # Add hidden layer
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization
        self.fc2 = nn.Linear(512, 256)  # Another hidden layer
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)  # Final output layer

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Replace the classifier head in ResNet with a custom head
model = models.resnet101(pretrained=True)
in_features = model.fc.in_features
model.fc = CustomClassifierHead(in_features, num_classes)
model = model.to(device)


# Hyperparameters
batch_size = 32
learning_rate = 3e-4
epochs = 30
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr = learning_rate)


# LoadDatalaoders

train_dataloader = DataLoader(train_dataset , batch_size = batch_size  , shuffle = True)
test_dataloader = DataLoader(test_dataset , batch_size = batch_size  , shuffle = False)



for epoch in range(epochs):
    model.train()
    
    training_loss = 0.0
    
    for image , label in train_dataloader:
        image , label = image.to(device) , label.to(device)
        
        y_pred = model(image)
        loss = criterion(y_pred , label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {training_loss/len(train_dataloader):.4f}')
        
    if(epoch%2 == 0):
        model.eval()
        
        test_corr = 0
        total = 0
        
        with torch.no_grad():
            for images , labels in test_dataloader:
                images , labels = images.to(device) , labels.to(device)
                
                y_pred = model(images)
                predicted = torch.max(y_pred.data , 1)[1]
                
                test_corr += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * test_corr / total
        print(f'Epoch [{epoch+1}/{epochs}], Test Accuracy: {accuracy:.2f}%')
        
        
torch.save(model, 'dog_breed_model_full.pth')
print("Entire model saved as 'dog_breed_model_full.pth'")
    