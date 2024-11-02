import gc
import matplotlib
matplotlib.use('TkAgg')  # GUI backend support
import matplotlib.pyplot as plt
import os
import pickle
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split, KFold

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'

# Load tokenized data
with open('tokenized_data.pkl', 'rb') as f:
    input_ids, attention_masks, labels = pickle.load(f)
print("Data loaded")

# K-Fold Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, val_index in kf.split(input_ids):
    print(f"Training Fold {fold}...")

    # Data Preparation
    train_inputs, validation_inputs = tf.gather(input_ids, train_index), tf.gather(input_ids, val_index)
    train_masks, validation_masks = tf.gather(attention_masks, train_index), tf.gather(attention_masks, val_index)
    train_labels, validation_labels = tf.gather(labels, train_index), tf.gather(labels, val_index)

    # Load Model
    if fold == 1:
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Train
    history = model.fit(
        [train_inputs, train_masks],
        train_labels,
        batch_size=16,
        epochs=4,
        validation_data=([validation_inputs, validation_masks], validation_labels)
    )

    # Free Resources
    tf.keras.backend.clear_session()
    gc.collect()

    fold += 1

# Final Training on All Data
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Train on all training data
history = model.fit(
    [input_ids, attention_masks],
    labels,
    batch_size=16,
    epochs=4
)

print("Training complete.")

# Save the model
tf.saved_model.save(model, "/home/lewis/proto/models/final_trained_model")
model.save_weights("/home/lewis/proto/models/final_trained_model_weights.h5")

# Save training history
import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

print("Training history saved.")
