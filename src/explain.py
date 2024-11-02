import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import numpy as np
import pandas as pd
import tensorflow as tf
from lime.lime_text import LimeTextExplainer
from transformers import BertTokenizer, TFBertForSequenceClassification
from proto.src.predata import load_and_preprocess_data

tf.keras.backend.clear_session()

# Load the model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# 确保加载的权重是针对 BERT 的
model.load_weights("/home/lewis/proto/models/final_trained_model_weights.h5")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Load your data here
data = load_and_preprocess_data()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Explanation with LIME
explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

# Choose an example to explain
example_text = data['text'].iloc[10]

def predict_fn(texts):
    tokens = tokenizer(texts, max_length=64, padding='max_length', truncation=True, return_tensors='tf')
    outputs = model(tokens, training=False)[0]  # Avoid gradient tracking
    return outputs.numpy()

# Reduce the number of features to mitigate memory usage
exp = explainer.explain_instance(example_text, predict_fn, num_features=5, num_samples=100)
exp.save_to_file("explanation.html")

tf.keras.backend.clear_session()  # Clear the session after usage
