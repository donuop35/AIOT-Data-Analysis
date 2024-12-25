# MNIST 手寫數字辨識
# 本程式使用 Dense Neural Network (DNN) 與 Convolutional Neural Network (CNN) 建立模型，進行手寫數字分類。

# 導入必要的套件
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. 載入資料
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. 資料處理
# 將圖片標準化到 [0, 1] 範圍內
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# 增加維度以適應 CNN 輸入結構 (批次大小, 高度, 寬度, 通道數)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# One-hot 編碼標籤
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. 建立 DNN 模型
dnn_model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練 DNN 模型
dnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 4. 建立 CNN 模型
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練 CNN 模型
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 5. 模型評估
# 使用測試資料進行預測
dnn_preds = dnn_model.predict(X_test)
cnn_preds = cnn_model.predict(X_test)

# 計算準確率
from sklearn.metrics import accuracy_score
print("DNN Accuracy:", accuracy_score(np.argmax(y_test, axis=1), np.argmax(dnn_preds, axis=1)))
print("CNN Accuracy:", accuracy_score(np.argmax(y_test, axis=1), np.argmax(cnn_preds, axis=1)))

# 繪製混淆矩陣
dnn_cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(dnn_preds, axis=1))
cnn_cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(cnn_preds, axis=1))

ConfusionMatrixDisplay(confusion_matrix=dnn_cm, display_labels=range(10)).plot()
plt.title("DNN Confusion Matrix")
plt.show()

ConfusionMatrixDisplay(confusion_matrix=cnn_cm, display_labels=range(10)).plot()
plt.title("CNN Confusion Matrix")
plt.show()
