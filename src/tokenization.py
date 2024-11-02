import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
from predata import load_and_preprocess_data
import pickle
from tqdm import tqdm

# Load and preprocess the data
print("Loading and preprocessing data...")
data = load_and_preprocess_data()
print("Data loaded and preprocessed!")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []
batch_size = 1000

# use tqdm to show the progress
print("Starting tokenization in batches...")
for start in tqdm(range(0, len(data), batch_size), desc="Tokenizing"):
    end = start + batch_size
    batch_texts = data['text'][start:end]
    encoded_batch = tokenizer(
        batch_texts.tolist(),
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    input_ids.append(encoded_batch['input_ids'])
    attention_masks.append(encoded_batch['attention_mask'])

print("Tokenization complete! Concatenating all batches...")

# Concatenate all batches
input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)
labels = tf.convert_to_tensor(data['label'].values)

# Save tokenized data
print("Saving tokenized data...")
with open('tokenized_data.pkl', 'wb') as f:
    pickle.dump((input_ids, attention_masks, labels), f)

print("Tokenization and saving complete!")
