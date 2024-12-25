# MNIST 手寫數字辨識 - PyTorch Lightning 實作
# 本程式使用 Dense Neural Network (DNN) 與 Convolutional Neural Network (CNN) 建立模型，進行手寫數字分類。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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
class DNN(LightningModule):
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
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# 3. 定義 CNN 模型
class CNN(LightningModule):
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

        # 自動計算 flatten_size
        dummy_input = torch.zeros(1, 1, 28, 28)  # MNIST 的輸入形狀
        conv_output = self.conv_layers(dummy_input)
        self.flatten_size = conv_output.view(-1).size(0)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# 訓練與驗證 DNN 模型
dnn_model = DNN()
dnn_trainer = Trainer(max_epochs=10, callbacks=[ModelCheckpoint(monitor='val_loss')])
dnn_trainer.fit(dnn_model, train_dataloaders=train_loader, val_dataloaders=test_loader)

# 訓練與驗證 CNN 模型
cnn_model = CNN()
cnn_trainer = Trainer(max_epochs=10, callbacks=[ModelCheckpoint(monitor='val_loss')])
cnn_trainer.fit(cnn_model, train_dataloaders=train_loader, val_dataloaders=test_loader)

# 評估模型
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    return all_labels, all_preds

print("Evaluating DNN...")
dnn_labels, dnn_preds = evaluate_model(dnn_model, test_loader)
print("DNN Accuracy:", accuracy_score(dnn_labels, dnn_preds))

print("Evaluating CNN...")
cnn_labels, cnn_preds = evaluate_model(cnn_model, test_loader)
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
