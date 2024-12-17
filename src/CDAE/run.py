import os
import pandas as pd
import torch
from .preprocessing import load_data, encode_data, generate_negative_samples, prepare_final_data
from .model import CDAEModel, train_model
from .inference import recommend_top_k

def main(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, args.dataset.data_path, 'train_ratings.csv')
    train_df = pd.read_csv(data_path)
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
                                               num_epochs=args.model_args.epochs,
                                               batch_size=args.model_args.batch_size,
                                               learning_rate=args.model_args.lr,
                                               device=device)

    recommendation_result = recommend_top_k(trained_model_instance,
                                            final_data_encoded,
                                            num_users=num_users,
                                            k=args.model_args.k)

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
