import pandas as pd
import os

def load_ratings_data(filepath='data/sample_ratings.csv'):
    """
    Load user-item rating data from a CSV file.

    Args:
        filepath (str): Path to the ratings CSV file.

    Returns:
        pd.DataFrame: DataFrame containing userId, itemId, and rating columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    required_cols = {'userId', 'itemId', 'rating'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    return df

if __name__ == "__main__":
    df = load_ratings_data()
    print(df.head())