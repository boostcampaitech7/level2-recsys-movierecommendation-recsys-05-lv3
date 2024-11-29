import os
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from preprocessing import load_data, encode_data, generate_negative_samples, prepare_final_data
from model import CDAEModel, train_model
from inference import recommend_top_k

def main(data_path):
    train_df = load_data(data_path)
    train_df_encoded, user_encoder, item_encoder = encode_data(train_df)

    num_users = train_df_encoded['user_id'].nunique()
    num_items = train_df_encoded['item_id'].nunique()

    negative_samples_df = generate_negative_samples(train_df_encoded.copy(), num_items)
    final_data_encoded = prepare_final_data(train_df_encoded.copy(), negative_samples_df)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_instance = CDAEModel(num_users=num_users,
                               num_items=num_items,
                               embedding_dim=128)

    trained_model_instance, _, _ = train_model(model_instance,
                                               final_data_encoded,
                                               num_items,
                                               num_epochs=10,
                                               batch_size=128,
                                               learning_rate=0.001,
                                               device=device)

    recommendation_result = recommend_top_k(trained_model_instance,
                                            final_data_encoded,
                                            num_users=num_users,
                                            k=10)

    header = True 
    chunk_size = 1000 
    rows = []

    for i, (user_id, item_ids) in enumerate(recommendation_result.items()):
        original_user_id = user_encoder.inverse_transform([user_id])[0]
        original_item_ids = item_encoder.inverse_transform(item_ids)

        for item_id in original_item_ids:
            rows.append([original_user_id, item_id]) 

        if (i + 1) % chunk_size == 0 or i == len(recommendation_result) - 1:
            df = pd.DataFrame(rows, columns=["user", "item"]) 
            df.to_csv('../../saved/CDAE.csv', mode='w', header=header, index=False) 
            header = False 
            rows = [] 

    print("추천 결과가 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, default="../../../data/train/train_ratings.csv")
    
    args = parser.parse_args()
    
    main(args.data_path)