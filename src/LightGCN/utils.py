import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, df, num_items):
        self.users = df['user'].values
        self.pos_items = df['item'].values
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        neg_item = np.random.randint(self.num_items)
        while neg_item in self.pos_items[self.users == user]:
            neg_item = np.random.randint(self.num_items)
        return user, pos_item, neg_item

def create_adj_matrix(df, num_users, num_items):
    user_nodes = df['user'].values
    item_nodes = df['item'].values + num_users
    edge_index = np.array([np.concatenate([user_nodes, item_nodes]),
                          np.concatenate([item_nodes, user_nodes])])
    adj = sp.coo_matrix((np.ones(len(edge_index[0])), 
                        (edge_index[0], edge_index[1])),
                       shape=(num_users + num_items, num_users + num_items),
                       dtype=np.float32)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    return torch.sparse.FloatTensor(
        torch.LongTensor(np.array([norm_adj.row, norm_adj.col])),
        torch.FloatTensor(norm_adj.data),
        torch.Size(norm_adj.shape)
    )

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
