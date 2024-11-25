import torch
import pandas as pd
import os
from train import get_predictions

def generate_recommendations(model, adj_matrix, num_users, num_items, user_interactions, device, k=10):
    model.eval()
    recommendations = {}
    
    predictions = get_predictions(model, adj_matrix, num_users, num_items, device, k)
    
    # 이미 본 영화 제외
    for user in range(num_users):
        watched_items = user_interactions.get(user, set())
        user_preds = predictions[user]
        filtered_preds = [item for item in user_preds if item not in watched_items]
        recommendations[user] = filtered_preds[:k]
    
    return recommendations

def create_submission(recommendations, user_id_map, item_id_map, output_path):
    users = []
    items = []
    
    original_user_ids = user_id_map['original_id'].tolist()
    original_item_ids = item_id_map['original_id'].tolist()
    
    for user, rec_items in recommendations.items():
        users.extend([original_user_ids[user]] * 10)
        items.extend([original_item_ids[item] for item in rec_items])
    
    submission_df = pd.DataFrame({
        'user': users,
        'item': items
    })
    
    submission_df.to_csv(output_path, index=False)