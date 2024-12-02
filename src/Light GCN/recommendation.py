import torch
import pandas as pd
import os

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