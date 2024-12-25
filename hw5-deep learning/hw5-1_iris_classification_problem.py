import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.tensorboard import SummaryWriter

# CRISP-DM Step 1: 商業理解
# 繁體中文說明
# 目標：通過機器學習模型分類 Iris 花卵的品種 (Setosa, Versicolor, Virginica)。應用場景包括植物分類、自動化農業等。
# 問題類型：多分類問題

# CRISP-DM Step 2: 資料理解
# 載入資料
from sklearn.datasets import load_iris
iris = load_iris()

# 創建 DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# 資料描述與統計
print("\n資料描述:")
print(iris.DESCR)
print("\n前五筆資料:")
print(iris_df.head())

# 資料可視化
sns.pairplot(iris_df, hue='species')
plt.show()

# CRISP-DM Step 3: 資料準備
# 資料分割與正規化
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# CRISP-DM Step 4: 建模
# (1) Scikit-learn 模型
# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# (2) TensorFlow/Keras 模型
# Dense NN 模型
def build_dense_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(4,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

keras_model = build_dense_model()
history = keras_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

# 評估
keras_acc = keras_model.evaluate(X_test, y_test, verbose=0)[1]

# 繪製 Keras 模型的訓練曲線
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Keras Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Keras Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# (3) PyTorch 模型
class IrisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = IrisDataset(X_train, y_train)
test_dataset = IrisDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = PyTorchModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

pytorch_losses = []

def train_model(model, loader, criterion, optimizer, epoch_losses):
    model.train()
    epoch_loss = 0
    for features, labels in loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_losses.append(epoch_loss / len(loader))

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 訓練
for epoch in range(50):
    train_model(model, train_loader, criterion, optimizer, pytorch_losses)

pytorch_acc = evaluate_model(model, test_loader)

# PyTorch 模型訓練損失圖
plt.figure(figsize=(6, 4))
plt.plot(pytorch_losses, label='Training Loss')
plt.title('PyTorch Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# (4) PyTorch Lightning 模型
class LightningModel(pl.LightningModule):
    def __init__(self):
        super(LightningModel, self).__init__()
        self.model = PyTorchModel()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

class LossCallback(pl.Callback):
    def __init__(self):
        self.losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        avg_loss = trainer.callback_metrics['train_loss'].item()
        self.losses.append(avg_loss)

loss_callback = LossCallback()
lightning_model = LightningModel()
trainer = Trainer(max_epochs=50, accelerator="cpu", callbacks=[loss_callback])
trainer.fit(lightning_model, train_loader, test_loader)
lightning_model.eval()

# CRISP-DM Step 5: 模型訓練與可視化
print("KNN Accuracy:", knn_acc)
print("Decision Tree Accuracy:", dt_acc)
print("Keras Accuracy:", keras_acc)
print("PyTorch Accuracy:", pytorch_acc)
print("PyTorch Lightning Accuracy:", trainer.callback_metrics['val_acc'].item())

# 模型效能比較
models = ['KNN', 'Decision Tree', 'Keras', 'PyTorch', 'PyTorch Lightning']
accuracies = [knn_acc, dt_acc, keras_acc, pytorch_acc, trainer.callback_metrics['val_acc'].item()]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.ylim(0, 1)
plt.show()

# CRISP-DM Step 6: 模型評估
# 混淆矩陣與分類報告
knn_cm = confusion_matrix(y_test, knn_pred)
dt_cm = confusion_matrix(y_test, dt_pred)
keras_pred = np.argmax(keras_model.predict(X_test), axis=1)
keras_cm = confusion_matrix(y_test, keras_pred)

# 修正 PyTorch 模型的混淆矩陣計算
pytorch_preds = []
pytorch_actuals = []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, preds = torch.max(outputs, 1)  # 對每個 batch 找到概率最大的類別
        pytorch_preds.extend(preds.numpy())  # 收集預測結果
        pytorch_actuals.extend(labels.numpy())  # 收集真實標籤

pytorch_cm = confusion_matrix(pytorch_actuals, pytorch_preds)

# 計算 PyTorch Lightning 模型混淆矩陣
lightning_preds = []
lightning_actuals = []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = lightning_model(features)
        _, preds = torch.max(outputs, 1)
        lightning_preds.extend(preds.numpy())
        lightning_actuals.extend(labels.numpy())
lightning_cm = confusion_matrix(lightning_actuals, lightning_preds)

# 可視化所有模型的混淆矩陣
plt.figure(figsize=(36, 6))

plt.subplot(1, 5, 1)
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Reds', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 5, 2)
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Greens', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 5, 3)
sns.heatmap(keras_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Keras Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 5, 4)
sns.heatmap(pytorch_cm, annot=True, fmt='d', cmap='Purples', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('PyTorch Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 5, 5)
sns.heatmap(lightning_cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('PyTorch Lightning Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()
