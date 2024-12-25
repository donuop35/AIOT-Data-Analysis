import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import Sequential, mixed_precision
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# CRISP-DM Step 1: 商業理解
# 繁體中文說明
# 目標：通過機器學習模型對 CIFAR-10 圖片進行分類，應用場景包括圖片識別、自動駕駛等。
# 問題類型：多分類問題

# CRISP-DM Step 2: 資料理解
# 資料處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 增加 num_workers 加速數據加載
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

# CRISP-DM Step 3: 資料準備
num_classes = 10
print("PyTorch Device:", torch.cuda.current_device(), torch.cuda.get_device_name())
# CRISP-DM Step 4: 建模
# (1) TensorFlow/Keras 模型
# 啟用混合精度訓練
mixed_precision.set_global_policy('mixed_float16')

# 強制 TensorFlow 使用 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 設定記憶體增長，避免占用過多 GPU 記憶體
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow is using GPU:", gpus)
    except RuntimeError as e:
        print(e)

def build_keras_model():
    base_model = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(32, 32, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', dtype='float32')  # 保持輸出為 float32
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

keras_model = build_keras_model()

# 使用 tf.data 進行數據加載以加速 GPU 運算
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# 將 PyTorch 的 CIFAR-10 資料集轉為 TensorFlow Dataset
train_dataset_tf = tf.data.Dataset.from_tensor_slices((train_dataset.data, train_dataset.targets))
train_dataset_tf = train_dataset_tf.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

test_dataset_tf = tf.data.Dataset.from_tensor_slices((test_dataset.data, test_dataset.targets))
test_dataset_tf = test_dataset_tf.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

# 確認 TensorFlow 使用的設備
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("TensorFlow 可用 GPU 數量:", len(physical_devices))
for i, device in enumerate(physical_devices):
    print(f"TensorFlow GPU {i}: {device}")

# Keras 模型訓練
history = keras_model.fit(
    train_dataset_tf,
    validation_data=test_dataset_tf,
    epochs=5,
    verbose=1
)

# 評估
keras_acc = keras_model.evaluate(test_dataset_tf, np.array(test_dataset.targets), verbose=0)[1]
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

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

# (2) PyTorch 模型
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.model = models.vgg19(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)

# 移動模型到 GPU 並確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
pytorch_model = PyTorchModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

pytorch_losses = []

# 啟用混合精度訓練
scaler = torch.cuda.amp.GradScaler()

def train_pytorch_model(model, loader, criterion, optimizer, epoch_losses):
    model.train()
    epoch_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)  # 移動數據到 GPU
        optimizer.zero_grad()

        # 混合精度訓練
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    epoch_losses.append(epoch_loss / len(loader))

# 確保 PyTorch 正在使用 GPU
print("PyTorch 使用的設備:", device)

# Reduce epochs to improve speed
for epoch in range(5):
    train_pytorch_model(pytorch_model.cuda(), train_loader, criterion, optimizer, pytorch_losses)

pytorch_model.eval()
pytorch_acc = sum((torch.argmax(pytorch_model(images.cuda()), dim=1) == labels.cuda()).sum().item()
                  for images, labels in test_loader) / len(test_dataset)
print("PyTorch is using device:", torch.cuda.current_device(), torch.cuda.get_device_name())

# PyTorch 模型訓練損失圖
plt.figure(figsize=(6, 4))
plt.plot(pytorch_losses, label='Training Loss')
plt.title('PyTorch Model Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# (3) PyTorch Lightning 模型
class LightningVGG19(pl.LightningModule):
    def __init__(self):
        super(LightningVGG19, self).__init__()
        self.model = models.vgg19(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        with torch.cuda.amp.autocast():  # 啟用混合精度
            outputs = self(images)
            loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# 確保 PyTorch Lightning 正確使用 GPU
print("PyTorch Lightning 是否使用 GPU:", torch.cuda.is_available())

lightning_model = LightningVGG19()
trainer = Trainer(max_epochs=5, accelerator="gpu", devices=1, precision=16)  # 啟用 16-bit 精度
trainer.fit(lightning_model, train_loader, test_loader)
lightning_model.eval()
lightning_acc = trainer.callback_metrics['val_acc'].item()

# CRISP-DM Step 5: 模型訓練與可視化
print("Keras Accuracy:", keras_acc)
print("PyTorch Accuracy:", pytorch_acc)
print("PyTorch Lightning Accuracy:", lightning_acc)

# 模型效能比較
models = ['Keras', 'PyTorch', 'PyTorch Lightning']
accuracies = [keras_acc, pytorch_acc, lightning_acc]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies, palette="viridis")
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.ylim(0, 1)
plt.show()

# CRISP-DM Step 6: 模型評估
# 混淆矩陣與分類報告
keras_preds = np.argmax(keras_model.predict(test_loader.dataset.data.astype('float32') / 255.0), axis=1)
keras_cm = confusion_matrix(np.array(test_dataset.targets), keras_preds)

pytorch_preds = []
pytorch_actuals = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()  # 確保數據移到 GPU
        outputs = pytorch_model(images)
        _, preds = torch.max(outputs, 1)
        pytorch_preds.extend(preds.cpu().numpy())  # 從 GPU 回到 CPU
        pytorch_actuals.extend(labels.cpu().numpy())

pytorch_cm = confusion_matrix(pytorch_actuals, pytorch_preds)

lightning_preds = []
lightning_actuals = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()  # 確保數據移到 GPU
        outputs = lightning_model(images)
        _, preds = torch.max(outputs, 1)
        lightning_preds.extend(preds.cpu().numpy())  # 從 GPU 回到 CPU
        lightning_actuals.extend(labels.cpu().numpy())

lightning_cm = confusion_matrix(lightning_actuals, lightning_preds)

# 可視化所有模型的混淆矩陣
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.heatmap(keras_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.title('Keras Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 3, 2)
sns.heatmap(pytorch_cm, annot=True, fmt='d', cmap='Purples', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.title('PyTorch Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 3, 3)
sns.heatmap(lightning_cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.title('PyTorch Lightning Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()
