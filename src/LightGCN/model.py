import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    """
    LightGCN (Light Graph Convolutional Network) 모델 클래스

    이 클래스는 He et al. (2020)의 논문 "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"에서 
    설명된 LightGCN 아키텍처를 구현합니다.

    매개변수:
        n_users (int): 데이터셋의 사용자 수
        n_items (int): 데이터셋의 아이템 수
        n_layers (int): Light Graph Convolution 레이어의 수
        embedding_dim (int): 사용자 및 아이템 임베딩의 차원
        lr (float, 선택적): L2 정규화를 위한 학습률. 기본값은 1e-5

    속성:
        user_embedding (nn.Embedding): 사용자를 위한 임베딩 레이어
        item_embedding (nn.Embedding): 아이템을 위한 임베딩 레이어

    메서드:
        forward(adj_matrix): 모델의 순전파를 수행합니다.
        bpr_loss(users, pos_items, neg_items): 학습을 위한 BPR 손실을 계산합니다.
    """
    def __init__(self, n_users, n_items, n_layers, embedding_dim, lr=1e-5):
        """LightGCN 모델을 초기화합니다."""
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.lr = lr

    def forward(self, adj_matrix):
        """
        LightGCN 모델의 순전파를 수행합니다.

        매개변수:
            adj_matrix (torch.Tensor): 사용자-아이템 상호작용 그래프의 인접 행렬

        반환값:
            tuple: 그래프 컨볼루션 후의 사용자 및 아이템 임베딩을 포함하는 튜플
        """
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings_list = [all_embeddings]
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        return torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

    def bpr_loss(self, users, pos_items, neg_items):
        """
        Bayesian Personalized Ranking (BPR) 손실을 계산합니다.

        매개변수:
            users (torch.Tensor): 사용자 인덱스의 텐서
            pos_items (torch.Tensor): 긍정적 아이템 인덱스의 텐서
            neg_items (torch.Tensor): 부정적 아이템 인덱스의 텐서

        반환값:
            torch.Tensor: L2 정규화가 포함된 계산된 BPR 손실
        """
        users_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        l2_reg = (users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / len(users)
        return loss + l2_reg * self.lr