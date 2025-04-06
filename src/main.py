import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_loader import load_ratings_data
from src.matrix_factorization import train_svd_model
from src.deep_learning_model import RatingsDataset, NeuralCF

def evaluate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        for users, items, ratings in dataloader:
            preds = model(users, items)
            print("Predictions:", preds[:5])
            print("Actuals:", ratings[:5])
            break

def run():
    df = load_ratings_data()

    # Matrix Factorization
    print("\nTraining SVD model...")
    svd_preds = train_svd_model(df)
    print("SVD Predictions:")
    print(svd_preds.head())

    # Deep Learning Model
    print("\nTraining Neural Collaborative Filtering model...")
    num_users = df['userId'].max()
    num_items = df['itemId'].max()
    dataset = RatingsDataset(df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = NeuralCF(num_users=num_users, num_items=num_items)
    evaluate_model(model, dataloader)

if __name__ == "__main__":
    run()
