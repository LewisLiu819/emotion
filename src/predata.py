# data_preprocessing.py
import pandas as pd
import re
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    # Load the Dataset
    data = pd.read_csv("/home/lewis/proto/data/sampled_dataset.csv", encoding="ISO-8859-1", names=["target", "id", "date", "flag", "user", "text"])
    # data = pd.read_csv("/home/lewis/proto/data/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", names=["target", "id", "date", "flag", "user", "text"])

    # 0: pos, 1:neg
    data['label'] = data['target'].apply(lambda x: 0 if x == 0 else 1)

    print("Starting to preprocess data...")

    # Data Preprocessing
    def preprocess_text(text):
        # remove URLs
        text = re.sub(r'http\S+', '', text)
        # remove signs
        text = re.sub(r'[^\w\s]', '', text)
        # lower case
        text = text.lower()
        return text

    data['text'] = data['text'].apply(preprocess_text)

    # Visualization
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='label', data=data)

    # Show specific values
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom')

    plt.title('Class Distribution')
    plt.xlabel('Emotion Class')
    plt.ylabel('Frequency')
    plt.show()

    print("Data Processed!")

    return data
