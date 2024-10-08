import os

# Đường dẫn đến thư mục chứa dữ liệu huấn luyện
train_dir = r'E:\\ANIMALS_DETECTION\\data\\train'

# Lấy danh sách các tên lớp và sắp xếp theo thứ tự
class_names = sorted(os.listdir(train_dir))

# In ra danh sách tên lớp theo chỉ số
for index, class_name in enumerate(class_names):
    print(f'Class {index}: {class_name}')
