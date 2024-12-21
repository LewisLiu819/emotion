import gc
import matplotlib
matplotlib.use('TkAgg')  # GUI backend support
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification
from sklearn.model_selection import KFold
import json
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Load tokenized data
with open('tokenized_data.pkl', 'rb') as f:
    input_ids, attention_masks, labels = pickle.load(f)
print("Data loaded")

train_ids, temp_ids, train_masks, temp_masks, train_labels, temp_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2, random_state=42)

val_ids, test_ids, val_masks, test_masks, val_labels, test_labels = train_test_split(
    temp_ids, temp_masks, temp_labels, test_size=0.5, random_state=42)

print("Data split into train, validation, and test sets.")

# Callback to track batch-wise metrics
class BatchMetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_accuracy = []
        self.batch_loss = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_accuracy.append(logs.get('accuracy'))
        self.batch_loss.append(logs.get('loss'))

# Adaptive Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    if epoch < 2:
        return lr
    return lr * 0.5

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
batch_logger = BatchMetricsLogger()

# Callback to track learning rates
class LRTracker(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        self.learning_rates.append(lr)

lr_tracker = LRTracker()

class LRBatchTracker(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_learning_rates = []

    def on_train_batch_end(self, batch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        self.batch_learning_rates.append(lr)

lr_batch_tracker = LRBatchTracker()

# Final Training on All Data
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(
    [train_ids, train_masks],
    train_labels,
    validation_data=([val_ids, val_masks], val_labels),
    batch_size=16,
    epochs=10,
    callbacks=[lr_callback, batch_logger, lr_batch_tracker]
)

print("Training complete.")

# Save the model
tf.saved_model.save(model, "/home/lewis/proto/models/final_trained_model")
model.save_weights("/home/lewis/proto/models/final_trained_model_weights.h5")

# Add learning rate and batch metrics to history
history.history['batch_lr'] = lr_batch_tracker.batch_learning_rates
history.history['batch_accuracy'] = batch_logger.batch_accuracy
history.history['batch_loss'] = batch_logger.batch_loss
history.history['val_accuracy'] = history.history.get('val_accuracy', [])
history.history['val_loss'] = history.history.get('val_loss', [])

# 测试集评估
test_predictions = model.predict([test_ids, test_masks])
test_predicted_labels = tf.argmax(test_predictions.logits, axis=1)
test_accuracy = tf.reduce_mean(tf.cast(test_predicted_labels == test_labels, tf.float32)).numpy()

print(f"Test Accuracy: {test_accuracy:.4f}")
history.history['test_accuracy'] = [test_accuracy]


# Clean history data and save
history_cleaned = {
    k: [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
    for k, v in history.history.items()
}

with open('training_history.json', 'w') as f:
    json.dump(history_cleaned, f)
print("Training history saved.")
