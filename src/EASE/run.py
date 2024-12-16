import os
import argparse
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import torch

from .model import *


def main(args):
    print("Load Data-----------------------------------")
    print(args)

    train = pd.read_csv(f'{args.dataset.data_path}/train_ratings.csv')
    user_list = train['user'].unique()
    item_list = train['item'].unique()

    user_to_index = {user: index for index, user in enumerate(user_list)}
    item_to_index = {item: index for index, item in enumerate(item_list)}

    train['user'] = train['user'].map(user_to_index)
    train['item'] = train['item'].map(item_to_index)

    num_users = len(user_to_index)
    num_items = len(item_to_index)

    rows = train['user'].values
    cols = train['item'].values


    data = np.full(len(train), 1)
    rating_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))


    model = EASE(args.model_args._lambda)
    print("Lets Data-----------------------------------")
    model.train(rating_matrix)

    print("Lets Inference-----------------------------------")
    predictions = model.predict(rating_matrix)
    N = 10
    top_n_items_per_user = []

    predictions[rating_matrix.nonzero()] = -np.inf
    for user_idx in range(predictions.shape[0]):
        user_predictions = predictions[user_idx, :]
        top_n_indices = np.argpartition(user_predictions, -N)[-N:]
        top_n_indices_sorted = top_n_indices[np.argsort(user_predictions[top_n_indices])[::-1]]
        top_n_items_per_user.append(top_n_indices_sorted)


    index_to_item = {index: item for item, index in item_to_index.items()}
    top_n_items_per_user_ids = [[index_to_item[idx] for idx in user_items] for user_items in top_n_items_per_user]

    result = []
    for user_id, items in zip(user_list, top_n_items_per_user_ids):
        for item_id in items:
            result.append((user_id, item_id))

    pro_dir = os.path.join(f'{args.dataset.save_path}')
    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)
    submission_df = pd.DataFrame(result, columns=['user', 'item'])
    submission_df.to_csv(f'{args.dataset.save_path}/EASE.csv', index=False)



