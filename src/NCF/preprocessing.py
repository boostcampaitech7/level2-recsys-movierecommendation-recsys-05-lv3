import pandas as pd
import numpy as np

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

def preprocess_data(df):
    user_mapping = {id: idx for idx, id in enumerate(df['user'].unique())}
    item_mapping = {id: idx for idx, id in enumerate(df['item'].unique())}
    df['user_id'] = df['user'].map(user_mapping)
    df['item_id'] = df['item'].map(item_mapping)
    df['interaction'] = 1
    return df, user_mapping, item_mapping

def negative_sampling(df, num_items, num_negatives=4):
    user_group = df.groupby('user_id')
    all_items = set(range(num_items))

    negative_samples = []
    for user, group in user_group:
        positive_items = set(group['item_id'])
        negative_candidates = list(all_items - positive_items)
        if len(negative_candidates) < num_negatives:
            num_negatives = len(negative_candidates)
        negative_items = np.random.choice(negative_candidates, size=num_negatives, replace=False)

        for item in negative_items:
            negative_samples.append([user, item, 0])

    negative_df = pd.DataFrame(negative_samples, columns=['user_id', 'item_id', 'interaction'])
    return negative_df