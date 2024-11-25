import torch
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import os
from model import LightGCN
from utils import create_adj_matrix, TrainDataset, calculate_recall_at_k
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_predictions(model, adj_matrix, num_users, num_items, device, k=10):
    model.eval()
    predictions = {}
    with torch.no_grad():
        user_emb, item_emb = model(adj_matrix)
        for user in range(num_users):
            user_vector = user_emb[user].unsqueeze(0)
            scores = torch.mm(user_vector, item_emb.t()).squeeze()
            _, top_items = torch.topk(scores, k=k)
            predictions[user] = top_items.tolist()
    return predictions

def train_model(train_data, val_data=None, n_layers=3, embedding_dim=64, batch_size=2048, num_epochs=10, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = pd.concat([train_data, val_data]) if val_data is not None else train_data
    model_name = f'lgcn_ly{n_layers}_ed{embedding_dim}_bs{batch_size}_ep{num_epochs}'

    num_users = data['user'].nunique()
    num_items = data['item'].nunique()
    adj_matrix = create_adj_matrix(data, num_users, num_items).to(device)
    
    model = LightGCN(num_users, num_items, n_layers, embedding_dim).to(device)
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

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, 
                       desc=f"Epoch {epoch+1}/{num_epochs}",
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
        print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
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

# if __name__ == "__main__":
#     base_path = '/data/ephemeral/home/ryu/data/'
    
#     # 데이터 로드
#     train_data = pd.read_csv(f'{base_path}/processed/processed_train_data.csv')
#     val_data = pd.read_csv(f'{base_path}/processed/processed_val_data.csv')
    
#     # 전체 데이터로 학습
#     model, adj_matrix = train_model(
#         train_data=train_data,
#         val_data=val_data,
#         n_layers=3,  # 원하는 파라미터 설정
#         embedding_dim=64,
#         batch_size=8192,
#         num_epochs=100,
#         model_name='final_model'
#     )