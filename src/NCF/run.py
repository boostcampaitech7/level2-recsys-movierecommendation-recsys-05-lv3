import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .preprocessing import preprocess_data, negative_sampling, create_interaction_matrix, InteractionDataset
from .model import NCF
from .train import train_model

def recommend_all_users(model, interaction_matrix, user_mapping, item_mapping, top_k=10, device='cpu'):
    model.eval()
    recommendations = []

    num_users = interaction_matrix.shape[0]
    for user_id in range(num_users):
        user_items = interaction_matrix[user_id].nonzero()[1]
        candidates = np.setdiff1d(np.arange(interaction_matrix.shape[1]), user_items)

        user_tensor = torch.tensor([user_id] * len(candidates)).to(device)
        item_tensor = torch.tensor(candidates).to(device)

        with torch.no_grad():
            scores = model(user_tensor, item_tensor).cpu().numpy()
        top_items = candidates[np.argsort(scores)[::-1][:top_k]]

        for item_id in top_items:
            recommendations.append((user_id, item_id))

    reverse_user_mapping = {v: k for k, v in user_mapping.items()}
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}
    recommendations = [
        (reverse_user_mapping[user], reverse_item_mapping[item])
        for user, item in recommendations
    ]

    recommendation_df = pd.DataFrame(recommendations, columns=['user', 'item'])
    return recommendation_df

def main(args):

    file_path = args.dataset.data_path+"train_ratings.csv"
    df = pd.read_csv(file_path)

    print("Preprocess data--------------")
    # Preprocess data
    df, user_mapping, item_mapping = preprocess_data(df)
    num_users = len(user_mapping)
    num_items = len(item_mapping)

    print("Negative Sampling--------------")
    # Negative Sampling
    negative_df = negative_sampling(df, num_items, num_negatives=args.model_args.num_negatives)
    train_df = pd.concat([df, negative_df])

    print("Create DataLoader--------------")
    # Create DataLoader
    train_dataset = InteractionDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=args.model_args.batch_size, shuffle=True)

    print("Initialize and Train Model--------------")
    # Initialize and Train Model
    model = NCF(num_users, num_items, embed_dim=args.model_args.embed_dim, hidden_dim=args.model_args.hidden_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(model, train_loader, epochs=args.model_args.epochs, lr=args.model_args.lr, device=device)

    print("Generate Recommendations--------------")
    # Generate Recommendations
    interaction_matrix = create_interaction_matrix(df, num_users, num_items)
    recommendation_df = recommend_all_users(model, interaction_matrix, user_mapping, item_mapping, top_k=args.model_args.top_k, device=device)

    # Save Recommendations
    output_dir = os.path.dirname(args.dataset.output_path)
    os.makedirs(output_dir, exist_ok=True)
    recommendation_df.to_csv((args.dataset.output_path+"NCF.csv"), index=False)
    print(f"Recommendations saved to {args.dataset.output_path+"NCF.csv"}")