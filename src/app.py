import gc
import matplotlib
matplotlib.use('TkAgg')  # GUI backend support
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import lime
import lime.lime_text
import numpy as np

# Load the model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_weights("/home/lewis/proto/models/final_trained_model_weights.h5")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# model.eval()

lime_explainer = lime.lime_text.LimeTextExplainer(class_names=[ "Negative", "Positive"])

def analyze_text(text):
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    outputs = model(inputs)
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=-1)

    sentiment = tf.argmax(probabilities, axis=1).numpy()[0]
    confidence = probabilities[0, sentiment].numpy()
    
    def predict_proba(texts):
        tokens = tokenizer(texts, return_tensors='tf', padding=True, truncation=True)
        outputs = model(tokens)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        return probs.numpy()
    
    explanation = lime_explainer.explain_instance(text, predict_proba, num_features=6)
    
    return sentiment, confidence, explanation

if __name__ == "__main__":
    user_text = input("请输入文本进行情感分析：")
    sentiment, confidence, explanation = analyze_text(user_text)

    sentiment_label = "Positive" if sentiment == 1 else "Negative"
    print(f"情感预测: {sentiment_label}")
    print(f"置信程度: {confidence:.2f}")

    print("LIME解释:")
    for feature, weight in explanation.as_list():
        print(f"Feature: {feature}, Weight: {weight}")
    explanation.save_to_file("explanation.html")