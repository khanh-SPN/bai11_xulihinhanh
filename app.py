import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input

# Đường dẫn thư mục
train_dir = r'Train'
val_dir = r'Validation'

# 1. Tiền xử lý dữ liệu
def load_data(directory):
    labels = []
    data = []
    label_map = {'dog': 0, 'cat': 1}  # Gán nhãn cho chó và mèo

    for label in ['dog', 'cat']:
        folder_path = os.path.join(directory, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:  # Kiểm tra xem ảnh có hợp lệ không
                img = cv2.resize(img, (64, 64))  # Resize ảnh về kích thước chuẩn
                data.append(img)
                labels.append(label_map[label])

    return np.array(data), np.array(labels)

# Load dữ liệu
X_train, y_train = load_data(train_dir)
X_val, y_val = load_data(val_dir)

# Chuẩn hóa dữ liệu
X_train = X_train / 255.0
X_val = X_val / 255.0

# 2. Xây dựng mô hình CNN
cnn_model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile mô hình CNN
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Huấn luyện mô hình CNN
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Đánh giá CNN
loss, accuracy = cnn_model.evaluate(X_val, y_val)
print("Độ chính xác của CNN:", accuracy)

# 4. Sử dụng mô hình CNN để dự đoán
sample_img = X_val[0]  # Lấy một ảnh mẫu để dự đoán
sample_label = y_val[0]
pred_cnn = np.argmax(cnn_model.predict(sample_img.reshape(1, 64, 64, 3)), axis=1)
print(f"Giá trị thực tế: {sample_label}")
print(f"Dự đoán từ CNN: {pred_cnn[0]}")

# 5. Giả lập R-CNN bằng cách tạo các vùng quan tâm đơn giản
def simple_region_proposals(image):
    h, w, _ = image.shape
    # Chia ảnh thành 4 vùng bằng nhau
    regions = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w // 2, h // 2),
        (0, h // 2, w // 2, h // 2),
        (w // 2, h // 2, w // 2, h // 2)
    ]
    return regions

# Sử dụng phương pháp đơn giản để tạo các vùng quan tâm
regions = simple_region_proposals((sample_img * 255).astype(np.uint8))
print("Số vùng quan tâm đơn giản được tạo:", len(regions))
for i, (x, y, w, h) in enumerate(regions):
    print(f"Vùng {i+1}: (x={x}, y={y}, w={w}, h={h})")
