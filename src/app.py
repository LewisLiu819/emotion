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

def normalize_weights_with_sign(explanation):
    """
    Normalize weights to the range [-1, -0.1] for negative weights and [0.1, 1] for positive weights.
    """
    weights = [weight for _, weight in explanation.as_list()]
    positive_weights = [weight for weight in weights if weight > 0]
    negative_weights = [weight for weight in weights if weight < 0]

    # Normalization for positive weights
    if positive_weights:
        min_positive = min(positive_weights)
        max_positive = max(positive_weights)
    else:
        min_positive = max_positive = 0

    # Normalization for negative weights
    if negative_weights:
        min_negative = min(negative_weights)
        max_negative = max(negative_weights)
    else:
        min_negative = max_negative = 0

    normalized_features = []
    for feature, weight in explanation.as_list():
        if weight > 0:  # Normalize positive weights
            normalized_weight = (
                (weight - min_positive) / (max_positive - min_positive) * (1 - 0.1) + 0.1
                if max_positive != min_positive else 0.1
            )
        elif weight < 0:  # Normalize negative weights
            normalized_weight = (
                (weight - min_negative) / (max_negative - min_negative) * (-0.1 + 1) - 1
                if max_negative != min_negative else -0.1
            )
        else:
            normalized_weight = 0  # Handle zero weight (rare case)

        normalized_features.append((feature, normalized_weight))

    return normalized_features


def save_visualized_html_with_normalized_weights_signed(explanation, filename="normalized_signed_explanation.html"):
    """
    Save LIME explanation as an HTML file with signed normalized weights for better visualization.
    """
    # Normalize weights with signed ranges
    normalized_features = normalize_weights_with_sign(explanation)

    # Create a custom HTML visualization for the explanation
    html_content = """
    <html>
    <head>
        <title>LIME Explanation (Signed Normalized)</title>
    </head>
    <body>
        <h1>LIME Explanation with Signed Normalized Weights</h1>
        <table border="1" style="border-collapse: collapse; width: 50%;">
            <tr>
                <th>Feature</th>
                <th>Signed Normalized Weight</th>
                <th>Bar Chart</th>
            </tr>
    """

    for feature, normalized_weight in normalized_features:
        bar_width = int(abs(normalized_weight) * 100)  # Scale bar width for visualization
        bar_color = "green" if normalized_weight > 0 else "red"
        html_content += f"""
        <tr>
            <td>{feature}</td>
            <td>{normalized_weight:.2f}</td>
            <td>
                <div style="width: {bar_width}px; height: 20px; background-color: {bar_color};"></div>
            </td>
        </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    # Save the custom HTML content to a file
    with open(filename, "w") as file:
        file.write(html_content)
    print(f"Signed normalized LIME explanation saved to {filename}")


# if __name__ == "__main__":
#     user_text = input("请输入文本进行情感分析：")
#     sentiment, confidence, explanation = analyze_text(user_text)

#     sentiment_label = "Positive" if sentiment == 1 else "Negative"
#     print(f"情感预测: {sentiment_label}")
#     print(f"置信程度: {confidence:.2f}")

#     # 保存带归一化权重（支持正负）的 LIME 可视化 HTML 文件
#     save_visualized_html_with_normalized_weights_signed(explanation)


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