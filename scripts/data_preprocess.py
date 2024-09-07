import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_file, output_dir):
    df = pd.read_csv(input_file)
    train, val_test = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, random_state=42)
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

if __name__ == "__main__":
    preprocess_data('data/raw/articles.csv', 'data/processed')
