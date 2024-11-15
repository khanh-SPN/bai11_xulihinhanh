import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
import matplotlib.pyplot as plt

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

# 5. Giả lập R-CNN sử dụng Selective Search (nếu ximgproc không khả dụng, hãy dùng phương pháp khác)
def selective_search_rcnn(image):
    ss = cv2.ximgproc.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects[:100]  # Chọn 100 vùng đầu tiên

# Kiểm tra nếu ximgproc có khả dụng
try:
    import cv2.ximgproc  # Kiểm tra import
    print("ximgproc module is available.")
    
    # Chạy Selective Search trên ảnh mẫu
    rects = selective_search_rcnn((sample_img * 255).astype(np.uint8))

    # Hiển thị các vùng quan tâm
    sample_img_copy = (sample_img * 255).astype(np.uint8).copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(sample_img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(sample_img_copy, cv2.COLOR_BGR2RGB))
    plt.title("Vùng quan tâm (R-CNN giả lập)")
    plt.show()

except ImportError:
    print("Mô-đun ximgproc không khả dụng. Không thể thực hiện Selective Search.")

# Dự đoán cuối cùng (nếu ximgproc không khả dụng, bạn có thể bỏ qua phần này)
