import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from model import LightGCN
import os
import scipy.sparse as sp
from tqdm import tqdm

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

    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape = torch.Size(norm_adj.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)

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

def train(model, train_loader, optimizer, adj_matrix, device):
    model.train()
    total_loss = 0
    for users, pos_items, neg_items in tqdm(train_loader, desc="Training", leave=False):
        users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
        optimizer.zero_grad()
        loss = model.bpr_loss(users, pos_items, neg_items)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_path = '/data/ephemeral/home/ryu/data/'
    train_data = pd.read_csv(os.path.join(base_path, 'processed', 'processed_train_data.csv'))
    
    num_users = train_data['user'].nunique()
    num_items = train_data['item'].nunique()
    
    adj_matrix = create_adj_matrix(train_data, num_users, num_items).to(device)
    
    model = LightGCN(num_users, num_items, n_layers=3, embedding_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_dataset = TrainDataset(train_data, num_items)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    num_epochs = 50
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        loss = train(model, train_loader, optimizer, adj_matrix, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    torch.save(model.state_dict(), os.path.join(base_path, 'model.pth'))
    print(f"Model saved to {os.path.join(base_path, 'model.pth')}")

if __name__ == "__main__":
    main()