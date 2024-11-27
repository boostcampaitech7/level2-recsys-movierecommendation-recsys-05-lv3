import torch
import pandas as pd
import os
from train import train_model, get_predictions
from recommendation import create_submission

def main():
    base_path = '/data/ephemeral/home/ryu/data/'
    
    # 데이터 로드
    train_data = pd.read_csv(f'{base_path}/processed/processed_train_data.csv')
    val_data = pd.read_csv(f'{base_path}/processed/processed_val_data.csv')
    full_data = pd.concat([train_data, val_data])
    user_id_map = pd.read_csv(f'{base_path}/processed/user_id_map.csv')
    item_id_map = pd.read_csv(f'{base_path}/processed/item_id_map.csv')
    
    # 모델 학습
    model, adj_matrix, model_name = train_model(
        # 최종 학습시에는 
        # train_data=full_data,
        # val_data는 주석 처리
        train_data=full_data,
        # val_data=val_data,
        n_layers=3,
        embedding_dim=128,
        batch_size=2048,
        num_epochs=30
    )
    
    # 사용자별 시청 기록 생성
    # full_data = pd.concat([train_data, val_data])
    user_interactions = full_data.groupby('user')['item'].apply(set).to_dict()
    
    # 추천 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recommendations = get_predictions(
        model, 
        adj_matrix, 
        len(user_id_map), 
        len(item_id_map), 
        user_interactions,
        device
    )
    
    # 제출 파일 생성
    submission_path = f'{base_path}/output/{model_name}_submission.csv'
    create_submission(recommendations, user_id_map, item_id_map, submission_path)
    print(f"Submission file created: {submission_path}")

if __name__ == "__main__":
    main()