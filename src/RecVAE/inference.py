# inference.py

import numpy as np
import torch
from tqdm import tqdm

def recommend_top_k(model, train_data, device, k=10, batch_size=500):
    """
    Generate top-k recommended items for each user based on the trained model.
    
    Args:
        model: Trained VAE model.
        train_data: User-item interaction matrix (CSR format).
        device: Device to perform computation on (CPU or GPU).
        k: Number of items to recommend.
        batch_size: Batch size for processing users.
    
    Returns:
        recommendations: List of top-k recommended items for each user.
    """
    model.eval() 
    num_users = train_data.shape[0]
    recommendations = []

    for start_idx in tqdm(range(0, num_users, batch_size), desc="Generating Recommendations"):
        end_idx = min(start_idx + batch_size, num_users)
        batch_users = np.arange(start_idx, end_idx)

        ratings_in = torch.Tensor(train_data[batch_users].toarray()).to(device)

        with torch.no_grad():
            ratings_pred = model(ratings_in, calculate_loss=False).cpu().numpy()

        ratings_pred[train_data[batch_users].nonzero()] = -np.inf

        top_k_items = np.argpartition(-ratings_pred, k, axis=1)[:, :k]
        top_k_sorted = np.argsort(-ratings_pred[np.arange(ratings_pred.shape[0])[:, None], top_k_items], axis=1)
        top_k_items_sorted = top_k_items[np.arange(ratings_pred.shape[0])[:, None], top_k_sorted]

        recommendations.extend(top_k_items_sorted.tolist())

    return recommendations