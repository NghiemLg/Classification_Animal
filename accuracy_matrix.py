# plot_accuracy.py

import matplotlib.pyplot as plt
import pickle

# Nạp lại lịch sử huấn luyện từ file 'history.pkl'
with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

def plot_accuracy(history):
    """
    Vẽ biểu đồ độ chính xác (accuracy) và validation accuracy từ lịch sử huấn luyện của mô hình.
    """
    # Vẽ đồ thị accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Gọi hàm vẽ đồ thị
if __name__ == "__main__":
    plot_accuracy(history)
