import pandas as pd

data = pd.read_csv("data/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1")

sample_data = data.sample(n=100000, random_state=42)

sample_data.to_csv('data/sampled_dataset.csv', index=False)
