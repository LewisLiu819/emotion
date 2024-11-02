import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from transformers import BertTokenizer, TFBertForSequenceClassification
from predata import load_and_preprocess_data
import pickle

# Load the trained model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_weights("/home/lewis/proto/models/final_trained_model_weights.h5")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-8), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Load and preprocess the data
with open('tokenized_data.pkl', 'rb') as f:
    val_input_ids, val_attention_masks, val_labels = pickle.load(f)
print("Data loaded")

# # Tokenization and Input Preparation
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# val_input_ids = []
# val_attention_masks = []
# val_labels = data['label'].values

# for text in data['text']:
#     encoded_dict = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=128,
#         padding='max_length',
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors='tf'
#     )
#     val_input_ids.append(encoded_dict['input_ids'])
#     val_attention_masks.append(encoded_dict['attention_mask'])

# val_input_ids = tf.concat(val_input_ids, axis=0)
# val_attention_masks = tf.concat(val_attention_masks, axis=0)

# Final Evaluation on the validation set
eval_result = model.evaluate([val_input_ids, val_attention_masks], val_labels)
print(f"Validation Accuracy: {eval_result[1]:.2f}")

# Predict on validation set
predictions = model.predict([val_input_ids, val_attention_masks]).logits
predicted_labels = tf.argmax(predictions, axis=1).numpy()

# Calculate evaluation metrics
f1 = f1_score(val_labels, predicted_labels, average=None)
positive_f1 = f1[1]
negative_f1 = f1[0]
precision = precision_score(val_labels, predicted_labels, average=None)
positive_precision = precision[1]
negative_precision = precision[0]
recall = recall_score(val_labels, predicted_labels, average=None)
positive_recall = recall[1]
negative_recall = recall[0]
conf_matrix = confusion_matrix(val_labels, predicted_labels)

# Print the evaluation metrics
print(f"Positive F1 Score: {positive_f1*100:.2f}")
print(f"Negative F1 Score: {negative_f1*100:.2f}")
print(f"Positive Precision: {positive_precision*100:.2f}")
print(f"Negative Precision: {negative_precision*100:.2f}")
print(f"Positive Recall: {positive_recall*100:.2f}")
print(f"Negative Recall: {negative_recall*100:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Create a DataFrame to store results
results_df = pd.DataFrame({
    'Metric': ['Validation Accuracy', 'F1 Score', 'Precision', 'Recall'],
    'Value': [eval_result[1], f1, precision, recall]
})

# Save the evaluation results to a CSV file
results_df.to_csv("evaluation_results.csv", index=False)
print(f"Evaluation results saved to 'evaluation_results.csv'.")

# Save the confusion matrix to a CSV file
conf_matrix_df = pd.DataFrame(conf_matrix, 
                              index=['True Positive', 'True Negative'], 
                              columns=['Predicted Positive', 'Predicted Negative'])
conf_matrix_df.to_csv("confusion_matrix.csv", index=True)
print(f"Confusion matrix saved to 'confusion_matrix.csv'.")
