import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class RatingsDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = torch.tensor(ratings_df['userId'].values, dtype=torch.long)
        self.items = torch.tensor(ratings_df['itemId'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NeuralCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        x = torch.cat([user_vec, item_vec], dim=-1)
        return self.fc_layers(x).squeeze()

if __name__ == "__main__":
    from src.data_loader import load_ratings_data

    df = load_ratings_data()
    num_users = df['userId'].max()
    num_items = df['itemId'].max()

    dataset = RatingsDataset(df)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = NeuralCF(num_users=num_users, num_items=num_items)
    for users, items, ratings in dataloader:
        preds = model(users, items)
        print(preds[:5])
        break
