import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# CUDA 디버깅 활성화
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 1. 네거티브 샘플링 함수
def negative_sampling(df, num_items, num_negatives=4):
    user_group = df.groupby('user_id')
    all_items = set(range(num_items))  # 전체 아이템 ID (0부터 num_items-1까지)
    
    negative_samples = []
    
    for user, group in user_group:
        positive_items = set(group['item_id'])  # 해당 사용자의 이미 상호작용한 아이템
        negative_candidates = list(all_items - positive_items)  # 상호작용하지 않은 아이템
        if len(negative_candidates) < num_negatives:
            num_negatives = len(negative_candidates)  # 후보 아이템 부족 시 조정
        negative_items = np.random.choice(negative_candidates, size=num_negatives, replace=False)
        
        for item in negative_items:
            negative_samples.append([user, item, 0])  # interaction=0
    
    negative_df = pd.DataFrame(negative_samples, columns=['user_id', 'item_id', 'interaction'])
    return negative_df

# 2. InteractionDataset 클래스
class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = df['user_id'].values
        self.items = df['item_id'].values
        self.labels = df['interaction'].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

# 3. NCF 모델 정의
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=16, hidden_dim=64):
        super(NCF, self).__init__()
        
        # 공통 임베딩 레이어
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        # GMF 부분
        self.gmf_layer = nn.Sequential()
        
        # MLP 부분
        self.mlp_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embed_dim)  # GMF와 차원 일치
        )
        
        # GMF와 MLP 결합
        self.final_layer = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        # 임베딩 추출
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        # GMF: 요소별 곱
        gmf_output = user_embed * item_embed
        
        # MLP: 임베딩 결합 후 Fully Connected Network 처리
        mlp_input = torch.cat([user_embed, item_embed], dim=-1)
        mlp_output = self.mlp_layer(mlp_input)
        
        # GMF와 MLP 결합
        final_input = gmf_output + mlp_output  # 또는 torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.final_layer(final_input)
        return self.sigmoid(output).squeeze()


# 4. 학습 함수
def train_model(model, train_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (user, item, label) in enumerate(train_loader):
            user, item, label = user.to(device), item.to(device), label.to(device).float()
            
            optimizer.zero_grad()
            preds = model(user, item)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Epoch별 출력 (버퍼 플러시)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(train_loader):.4f}", flush=True)



# 5. 추천 함수
def recommend_all_users(model, interaction_matrix, user_mapping, item_mapping, top_k=10, device='cpu'):
    model.eval()
    recommendations = []

    num_users = interaction_matrix.shape[0]
    for user_id in range(num_users):
        user_items = interaction_matrix[user_id].nonzero()[1]
        candidates = np.setdiff1d(np.arange(interaction_matrix.shape[1]), user_items)
        
        user_tensor = torch.tensor([user_id] * len(candidates)).to(device)
        item_tensor = torch.tensor(candidates).to(device)
        
        with torch.no_grad():
            scores = model(user_tensor, item_tensor).cpu().numpy()
        top_items = candidates[np.argsort(scores)[::-1][:top_k]]
        
        for item_id in top_items:
            recommendations.append((user_id, item_id))
    
    reverse_user_mapping = {v: k for k, v in user_mapping.items()}
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}
    recommendations = [
        (reverse_user_mapping[user], reverse_item_mapping[item])
        for user, item in recommendations
    ]
    
    recommendation_df = pd.DataFrame(recommendations, columns=['user', 'item'])
    return recommendation_df

# 6. 메인 실행
if __name__ == "__main__":
    file_path = "data/train/train_ratings.csv"  # 데이터 경로
    df = pd.read_csv(file_path)

    # 사용자/아이템 ID 매핑
    user_mapping = {id: idx for idx, id in enumerate(df['user'].unique())}
    item_mapping = {id: idx for idx, id in enumerate(df['item'].unique())}
    df['user_id'] = df['user'].map(user_mapping)
    df['item_id'] = df['item'].map(item_mapping)
    df['interaction'] = 1  # 기본 interaction 값 (positive interactions)

    num_users = len(user_mapping)
    num_items = len(item_mapping)

    # 네거티브 샘플링 없이 데이터 사용
    train_dataset = InteractionDataset(df)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 모델 초기화 및 학습
    model = NCF(num_users, num_items, embed_dim=16, hidden_dim=64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(model, train_loader, epochs=10, lr=0.001, device=device)

    # 모든 사용자 추천 수행
    interaction_matrix = sp.lil_matrix((num_users, num_items))
    for _, row in df.iterrows():
        interaction_matrix[row['user_id'], row['item_id']] = 1
    top_k = 10
    recommendation_df = recommend_all_users(model, interaction_matrix, user_mapping, item_mapping, top_k=top_k, device=device)

    # 추천 결과 저장
    recommendation_df.to_csv("jw5.csv", index=False)
    print("추천 결과가 recommendations.csv 파일에 저장되었습니다.")
