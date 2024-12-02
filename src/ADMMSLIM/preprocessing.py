import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

def load_and_preprocess_data(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
    train_df['watched'] = 1
    train_df = train_df.drop(columns=['time'])

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    train_df['user_id'] = user_encoder.fit_transform(train_df['user'])
    train_df['item_id'] = item_encoder.fit_transform(train_df['item'])

    num_users = train_df['user_id'].nunique()
    num_items = train_df['item_id'].nunique()

    interactions = train_df['watched'].values
    user_item_matrix = csr_matrix(
        (interactions, (train_df['user_id'].values, train_df['item_id'].values)),
        shape=(num_users, num_items)
    )

    return train_df, user_item_matrix, user_encoder, item_encoder