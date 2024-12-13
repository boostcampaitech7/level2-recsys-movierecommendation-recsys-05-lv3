import pandas as pd
import numpy as np
import torch

def create_interaction_matrix(df, num_users, num_items):
    import scipy.sparse as sp
    interaction_matrix = sp.lil_matrix((num_users, num_items))
    for _, row in df.iterrows():
        interaction_matrix[row['user_id'], row['item_id']] = 1
    return interaction_matrix

class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.users = df['user_id'].values
        self.items = df['item_id'].values
        self.labels = df['interaction'].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]