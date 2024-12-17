import torch
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import os
from .model import LightGCN
from .utils import create_adj_matrix, TrainDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_predictions(model, adj_matrix, num_users, user_interactions, k=10):
    """
    모델을 사용하여 각 사용자에 대한 상위 k개의 아이템 추천을 생성합니다.

    매개변수:
    model (LightGCN): 학습된 LightGCN 모델
    adj_matrix (torch.Tensor): 정규화된 인접 행렬
    num_users (int): 전체 사용자 수
    user_interactions (dict): 각 사용자가 이미 상호작용한 아이템 집합
    k (int): 추천할 아이템 수 (기본값: 10)

    반환값:
    dict: 각 사용자에 대한 상위 k개 추천 아이템 목록
    """
    model.eval()
    predictions = {}
    with torch.no_grad():
        user_emb, item_emb = model(adj_matrix)
        for user in range(num_users):
            user_vector = user_emb[user].unsqueeze(0)
            scores = torch.mm(user_vector, item_emb.t()).squeeze()
            
            # 이미 시청한 아이템의 점수를 -inf로 설정
            watched_items = user_interactions.get(user, set())
            scores[list(watched_items)] = float('-inf')
            
            # 시청하지 않은 아이템 중에서 top-k 선택
            _, top_items = torch.topk(scores, k=k)
            predictions[user] = top_items.tolist()

    return predictions

def train_model(train_data, val_data=None, n_layers=3, embedding_dim=128, batch_size=2048, n_epochs=100, patience=5, lr=1e-5):
    """
    LightGCN 모델을 학습하고 최적의 모델을 반환합니다.

    매개변수:
    train_data (pd.DataFrame): 학습 데이터
    val_data (pd.DataFrame, optional): 검증 데이터 (기본값: None)
    n_layers (int): LightGCN 모델의 레이어 수 (기본값: 3)
    embedding_dim (int): 임베딩 차원 (기본값: 128)
    batch_size (int): 배치 크기 (기본값: 2048)
    n_epochs (int): 학습 에폭 수 (기본값: 100)
    patience (int): 조기 종료를 위한 인내심 (기본값: 5)
    lr (float): 학습률 (기본값: 1e-5)

    반환값:
    tuple: (학습된 모델, 인접 행렬, 모델 이름)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = pd.concat([train_data, val_data]) if val_data is not None else train_data
    model_name = f'lgcn_ly{n_layers}_ed{embedding_dim}_bs{batch_size}_ep{n_epochs}'

    num_users = data['user'].nunique()
    num_items = data['item'].nunique()
    adj_matrix = create_adj_matrix(data, num_users, num_items).to(device)
    
    model = LightGCN(num_users, num_items, n_layers, embedding_dim, lr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    train_loader = DataLoader(TrainDataset(data, num_items), 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4)
    
    best_loss = float('inf')
    no_improve = 0
    best_model_path = f'saved_models/{model_name}_best.pth'
    os.makedirs('saved_models', exist_ok=True)
    losses = []  # 에폭별 손실값을 저장할 리스트

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, 
                       desc=f"Epoch {epoch+1}/{n_epochs}",
                       position=0,
                       leave=True,
                       ncols=100)
        
        for users, pos_items, neg_items in progress_bar:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            optimizer.zero_grad()
            
            with autocast():
                loss = model.bpr_loss(users, pos_items, neg_items)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)  # 에폭별 손실값 저장
        print(f"\nEpoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Early stopping 체크
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'losses': losses,  # 손실값 히스토리도 저장
            }, best_model_path)
            print(f"Best model saved with loss: {best_loss:.4f}")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs")
            
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 최고 성능 모델 로드
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, adj_matrix, model_name
