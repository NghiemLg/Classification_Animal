import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

# Đường dẫn tới dữ liệu huấn luyện và kiểm thử
train_dir = 'data/train/'
validation_dir = 'data/validation/'

# Tạo pipeline để đọc và xử lý dữ liệu với augmentation cho tập train
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load dữ liệu từ thư mục, kích thước ảnh 150x150, batch size 32
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Xây dựng mô hình ResNet50V2 với pretrained weights trên ImageNet
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Đóng băng các lớp của mô hình ResNet để không huấn luyện lại chúng
base_model.trainable = False

# Thêm các lớp phía trên (custom head) để phù hợp với bài toán phân loại 8 lớp
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(8, activation='softmax')  # 8 lớp tương ứng với 8 loài động vật
])

# Compile mô hình
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    'animal_classification_model.keras',  # Đổi tên file để kết thúc bằng .keras
    monitor='val_loss',  # Theo dõi giá trị nào
    save_best_only=True,  # Chỉ lưu khi mô hình tốt hơn
    mode='min',  # Lưu khi giá trị theo dõi giảm
    verbose=1  # Hiển thị thông báo khi lưu
)


# Huấn luyện mô hình và lưu lại lịch sử huấn luyện
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint]  # Thêm callback vào đây
)

# Lưu lịch sử huấn luyện vào file 'history.pkl'
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Đánh giá mô hình
evaluation = model.evaluate(validation_generator)
print(f'Loss: {evaluation[0]}, Accuracy: {evaluation[1]}')
