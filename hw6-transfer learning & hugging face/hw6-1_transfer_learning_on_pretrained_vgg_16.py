import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

# 檢查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 設定資料路徑（已指定在 prompt 裡）
train_dir = "/content/Face-Mask-Detection-/facemask/train"
valid_dir = "/content/Face-Mask-Detection-/facemask/valid"

# 定義影像處理與資料增強
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),      # ResNet-18 可用 224x224
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])  # ImageNet 平均/標準差
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 建立資料集 (ImageFolder)
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)

# 建立 DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=32,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=32,
                                           shuffle=False)

# 取得類別索引與名稱
class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

# 載入 ResNet-18 預訓練模型 (ImageNet)
model = models.resnet18(pretrained=True)

# 凍結所有參數，不進行反向傳遞
for param in model.parameters():
    param.requires_grad = False

# 重新定義最後一層全連接層 (fc)
# 原本的輸出是 1000 類 (ImageNet)，現在改為 2 類
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# 移動到 GPU (若可用)
model = model.to(device)

# 定義損失函式與最佳化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

    num_epochs = 5  # 可自行調整
best_acc = 0.0

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    valid_loss, valid_acc = validate_one_epoch(model, valid_loader, criterion, device)

    # 若驗證集表現更好，則保存模型 (示範)
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), "best_resnet18_mask.pth")

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f} ")

test_url = "https://www.sayho.com/wp-content/uploads/2022/01/%E8%AA%AA%E5%A5%BD%E6%94%9D%E5%BD%B1-%E9%9F%93%E5%BC%8F%E8%AD%89%E4%BB%B6%E7%85%A7.jpg"
predict_image_from_url(test_url, model, device, class_names)
