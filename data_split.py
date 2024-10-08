import os
import shutil
import random

# Tạo thư mục train và validation
os.makedirs('data/train/', exist_ok=True)
os.makedirs('data/validation/', exist_ok=True)

# Tên các loài động vật
categories = ['butterfly', 'chicken', 'cow', 'dog', 'horse', 'sheep', 'spider', 'squirrel']

# Chia dữ liệu cho mỗi loài động vật
for category in categories:
    os.makedirs(f'data/train/{category}', exist_ok=True)
    os.makedirs(f'data/validation/{category}', exist_ok=True)

    # Đường dẫn đến thư mục chứa hình ảnh của loài động vật hiện tại
    category_path = f'data/{category}'
    images = os.listdir(category_path)
    
    # Xáo trộn ngẫu nhiên các hình ảnh
    random.shuffle(images)

    # Chia dữ liệu (80% train, 20% validation)
    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    validation_images = images[split_index:]

    # Di chuyển ảnh vào thư mục train
    for image in train_images:
        shutil.move(os.path.join(category_path, image), f'data/train/{category}/')

    # Di chuyển ảnh vào thư mục validation
    for image in validation_images:
        shutil.move(os.path.join(category_path, image), f'data/validation/{category}/')

    # Xóa thư mục gốc của loài động vật
