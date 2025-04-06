import numpy as np
from scipy.sparse.linalg import svds

def train_svd_model(ratings_df, k=20):
    """
    Train a matrix factorization model using truncated SVD.

    Args:
        ratings_df (pd.DataFrame): Input ratings data with userId, itemId, rating.
        k (int): Number of latent features.

    Returns:
        tuple: User-feature matrix, sigma, item-feature matrix transposed
    """
    user_item_matrix = ratings_df.pivot(index='userId', columns='itemId', values='rating').fillna(0)
    matrix = user_item_matrix.values
    user_ratings_mean = np.mean(matrix, axis=1)
    matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

    U, sigma, Vt = svds(matrix_demeaned, k=k)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

    return preds_df

if __name__ == "__main__":
    import pandas as pd
    from src.data_loader import load_ratings_data

    df = load_ratings_data()
    preds = train_svd_model(df)
    print(preds.head())
