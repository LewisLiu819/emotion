import json
import numpy as np
import matplotlib.pyplot as plt

# 加载训练历史
with open('training_history.json', 'r') as f:
    history = json.load(f)

# 修复数据类型
def fix_float32(data):
    if isinstance(data, list):
        return [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in data]
    return data

history_fixed = {k: fix_float32(v) for k, v in history.items()}

# 保存修复后的文件
with open('training_history_fixed.json', 'w') as f:
    json.dump(history_fixed, f)

print("Fixed training history saved to 'training_history_fixed.json'.")

# Plot learning rate trend
learning_rates = history_fixed.get('lr', [])
if learning_rates:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, marker='o', linestyle='-', label='Learning Rate')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.title('Learning Rate Trend', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("learning_rate_trend.png")
    plt.show()

# Plot batch-wise loss
batch_loss = history_fixed.get('batch_loss', [])
if batch_loss:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(batch_loss) + 1), batch_loss, color='blue', marker='o', linestyle='-', label='Batch Loss')
    plt.xlabel('Batches', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Batch-wise Loss', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("batch_loss_curve.png")
    plt.show()

# Plot batch-wise accuracy
batch_accuracy = history_fixed.get('batch_accuracy', [])
if batch_accuracy:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(batch_accuracy) + 1), batch_accuracy, color='green', marker='o', linestyle='-', label='Batch Accuracy')
    plt.xlabel('Batches', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Batch-wise Accuracy', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("batch_accuracy_curve.png")
    plt.show()

# Plot epoch-wise loss (if applicable)
loss = history_fixed.get('loss', [])
if loss:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss) + 1), loss, color='blue', marker='o', label='Training Loss')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Epoch-wise Loss', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("epoch_loss_curve.png")
    plt.show()

# Plot epoch-wise accuracy (if applicable)
accuracy = history_fixed.get('accuracy', [])
if accuracy:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(accuracy) + 1), accuracy, color='green', marker='o', label='Training Accuracy')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Epoch-wise Accuracy', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("epoch_accuracy_curve.png")
    plt.show()

# Plot batch-wise learning rate
batch_lr = history_fixed.get('batch_lr', [])
if batch_lr:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(batch_lr) + 1), batch_lr, color='purple', marker='o', linestyle='-', label='Batch Learning Rate')
    plt.xlabel('Batches', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.title('Batch-wise Learning Rate', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("batch_learning_rate_curve.png")
    plt.show()

# 绘制验证集准确率
val_accuracy = history_fixed.get('val_accuracy', [])
if val_accuracy:
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, color='blue', marker='o', label='Validation Accuracy')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Validation Accuracy Over Epochs', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("val_accuracy_curve.png")
    plt.show()

# 绘制测试集准确率
test_accuracy = history_fixed.get('test_accuracy', [])
if test_accuracy:
    plt.figure(figsize=(8, 6))
    plt.bar(['Test Accuracy'], test_accuracy, color='orange')
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Test Accuracy', fontsize=16)
    plt.tight_layout()
    plt.savefig("test_accuracy_bar.png")
    plt.show()
