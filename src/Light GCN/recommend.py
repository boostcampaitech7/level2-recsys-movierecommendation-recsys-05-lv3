import torch
import pandas as pd
import numpy as np
from model import LightGCN
from train import create_adj_matrix
import os

def generate_recommendations(model, adj_matrix, num_users, num_items, top_k=10):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(adj_matrix)
        
    recommendations = {}
    for user in range(num_users):
        user_vector = user_emb[user].unsqueeze(0)
        scores = torch.mm(user_vector, item_emb.t()).squeeze()
        _, top_items = torch.topk(scores, k=top_k)
        recommendations[user] = top_items.tolist()
    
    return recommendations

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_path = '/data/ephemeral/home/ryu/data/'
    train_data = pd.read_csv(os.path.join(base_path, 'processed', 'processed_train_data.csv'))
    
    num_users = train_data['user'].nunique()
    num_items = train_data['item'].nunique()
    
    adj_matrix = create_adj_matrix(train_data, num_users, num_items).to(device)
    
    model = LightGCN(num_users, num_items, n_layers=3, embedding_dim=64).to(device)
    model.load_state_dict(torch.load(os.path.join(base_path, 'model.pth')))
    
    recommendations = generate_recommendations(model, adj_matrix, num_users, num_items)
    
    user_id_map = pd.read_csv(os.path.join(base_path, 'processed', 'user_id_map.csv'))
    item_id_map = pd.read_csv(os.path.join(base_path, 'processed', 'item_id_map.csv'))
    
    original_user_ids = user_id_map['original_id'].tolist()
    original_item_ids = item_id_map['original_id'].tolist()
    
    users = []
    items = []
    for user, rec_items in recommendations.items():
        users.extend([original_user_ids[user]] * 10)
        items.extend([original_item_ids[item] for item in rec_items])
    
    submission_df = pd.DataFrame({
        'user': users,
        'item': items
    })
    
    submission_df.to_csv(os.path.join(base_path, 'output', 'lightgcn_submission.csv'), index=False)
    print(f"Recommendations saved to {os.path.join(base_path, 'output', 'lightgcn_submission.csv')}")

if __name__ == "__main__":
    main()