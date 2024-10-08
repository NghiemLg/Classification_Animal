from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load mô hình đã huấn luyện
model = load_model('animal_classification_model.keras')

# In ra kiến trúc mô hình
model.summary()

# Đường dẫn tới ảnh cần dự đoán
img_path = r'E:\ANIMALS_DETECTION\data\validation\squirrel\OIP-_lEB2E_BnSgc4Peh-rZ3rQHaFf.jpeg'

# Load ảnh và chuẩn bị
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Dự đoán
predictions = model.predict(img_array)
print("Predictions:", predictions)

# Tìm lớp có xác suất cao nhất
predicted_class_index = np.argmax(predictions, axis=1)

# Định nghĩa danh sách các lớp (tên lớp)
class_names = ['butterfly', 'chicken', 'cow', 'dao son', 'horse', 'sheep', 'spider', 'squirrel']


# Kiểm tra xem predicted_class_index có trong khoảng
if predicted_class_index.size > 0:
    predicted_label = class_names[predicted_class_index[0]]
    print("Predicted class:", predicted_label)

    # Hiển thị ảnh và chú thích
    plt.imshow(image.load_img(img_path))
    plt.title(f'Predicted: {predicted_label}')
    plt.axis('off')
    plt.show()
else:
    print("Không thể dự đoán lớp nào.")
