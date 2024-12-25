# MNIST 手寫數字辨識
# 本程式使用 Dense Neural Network (DNN) 與 Convolutional Neural Network (CNN) 建立模型，進行手寫數字分類。

# 導入必要的套件
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torchvision import datasets, transforms

# 1. 資料準備
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 定義 DNN 模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

dnn_model = DNN()

# 訓練 DNN
criterion = nn.CrossEntropyLoss()
dnn_optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)

def train_model(model, optimizer, criterion, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.view(inputs.size(0), -1))  # Flatten inputs for DNN
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.view(inputs.size(0), -1), targets  # 展平數據
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())  # 移回 CPU
            all_labels.extend(targets.cpu().numpy())  # 移回 CPU
    return all_labels, all_preds

print("Training DNN...")
train_model(dnn_model, dnn_optimizer, criterion, train_loader)

# 評估 DNN
print("Evaluating DNN...")
dnn_labels, dnn_preds = evaluate_model(dnn_model, test_loader)
print("DNN Accuracy:", accuracy_score(dnn_labels, dnn_preds))

# 3. 定義 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )
        # 計算卷積層輸出大小
        self.flatten_size = 64 * 4 * 4  # 64個通道，特徵圖大小為4x4
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

cnn_model = CNN()

# 訓練 CNN
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

def train_cnn_model(model, optimizer, criterion, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # No need to flatten inputs for CNN
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

def evaluate_cnn_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)  # CNN 不需要展平數據
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())  # 移回 CPU
            all_labels.extend(targets.cpu().numpy())  # 移回 CPU
    return all_labels, all_preds

print("Training CNN...")
train_cnn_model(cnn_model, cnn_optimizer, criterion, train_loader)

# 評估 CNN
print("Evaluating CNN...")
cnn_labels, cnn_preds = evaluate_cnn_model(cnn_model, test_loader)
print("CNN Accuracy:", accuracy_score(cnn_labels, cnn_preds))

# 繪製混淆矩陣
confusion_matrix_dnn = confusion_matrix(dnn_labels, dnn_preds)
confusion_matrix_cnn = confusion_matrix(cnn_labels, cnn_preds)

ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_dnn, display_labels=range(10)).plot()
plt.title("DNN Confusion Matrix")
plt.show()

ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_cnn, display_labels=range(10)).plot()
plt.title("CNN Confusion Matrix")
plt.show()
