import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, n_layers, embedding_dim):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, adj_matrix):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings_list = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        return torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

    def bpr_loss(self, users, pos_items, neg_items):
        users_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        l2_reg = (users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / len(users)
        return loss + l2_reg * 1e-5