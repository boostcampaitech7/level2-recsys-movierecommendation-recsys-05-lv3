import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .preprocessing import preprocess_data, negative_sampling, create_interaction_matrix, InteractionDataset
from .model import NCF
from .train import train_model

def recommend_all_users(model, interaction_matrix, user_mapping, item_mapping, top_k=10, device='cpu'):
    """
    주어진 모델을 사용하여 모든 사용자에 대해 추천 아이템을 생성합니다.

    Args:
        model (torch.nn.Module): 추천 모델 (예: NCF 모델).
        interaction_matrix (scipy.sparse.lil_matrix): 사용자-아이템 상호작용 행렬 (희소 행렬).
        user_mapping (dict): 사용자 ID를 인덱스로 매핑한 딕셔너리.
        item_mapping (dict): 아이템 ID를 인덱스로 매핑한 딕셔너리.
        top_k (int, optional): 각 사용자에 대해 추천할 상위 K개의 아이템. 기본값은 10.
        device (str, optional): 모델과 데이터를 배치할 디바이스 (예: 'cuda' 또는 'cpu'). 기본값은 'cpu'.

    Returns:
        pd.DataFrame: 추천된 사용자-아이템 쌍을 포함하는 데이터프레임.
    """
    model.eval()  # 모델을 평가 모드로 설정
    recommendations = []

    num_users = interaction_matrix.shape[0]  # 사용자 수
    for user_id in range(num_users):
        # 현재 사용자가 상호작용한 아이템들
        user_items = interaction_matrix[user_id].nonzero()[1]
        
        # 현재 사용자가 상호작용하지 않은 아이템들 (추천 후보)
        candidates = np.setdiff1d(np.arange(interaction_matrix.shape[1]), user_items)

        # 사용자와 후보 아이템에 대한 텐서 생성
        user_tensor = torch.tensor([user_id] * len(candidates)).to(device)
        item_tensor = torch.tensor(candidates).to(device)

        with torch.no_grad():
            # 모델을 통해 각 후보 아이템의 점수 예측
            scores = model(user_tensor, item_tensor).cpu().numpy()
        
        # 예측 점수가 높은 상위 K개 아이템을 선택
        top_items = candidates[np.argsort(scores)[::-1][:top_k]]

        # 추천된 아이템을 추천 리스트에 추가
        for item_id in top_items:
            recommendations.append((user_id, item_id))

    # 사용자와 아이템 ID를 원래의 ID로 변환 (매핑을 역으로 사용)
    reverse_user_mapping = {v: k for k, v in user_mapping.items()}
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}

    # 추천 결과를 원래의 사용자와 아이템 ID로 변환하여 리스트에 저장
    recommendations = [
        (reverse_user_mapping[user], reverse_item_mapping[item])
        for user, item in recommendations
    ]

    # 추천 결과를 데이터프레임 형식으로 변환
    recommendation_df = pd.DataFrame(recommendations, columns=['user', 'item'])
    
    return recommendation_df

def main(args):

    file_path = args.dataset.data_path+"train_ratings.csv"
    df = pd.read_csv(file_path)

    print("Preprocess data--------------")
    # Preprocess data
    df, user_mapping, item_mapping = preprocess_data(df)
    num_users = len(user_mapping)
    num_items = len(item_mapping)

    print("Negative Sampling--------------")
    # Negative Sampling
    negative_df = negative_sampling(df, num_items, num_negatives=args.model_args.num_negatives)
    train_df = pd.concat([df, negative_df])

    print("Create DataLoader--------------")
    # Create DataLoader
    train_dataset = InteractionDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=args.model_args.batch_size, shuffle=True)

    print("Initialize and Train Model--------------")
    # Initialize and Train Model
    model = NCF(num_users, num_items, embed_dim=args.model_args.embed_dim, hidden_dim=args.model_args.hidden_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(model, train_loader, epochs=args.model_args.epochs, lr=args.model_args.lr, device=device)

    print("Generate Recommendations--------------")
    # Generate Recommendations
    interaction_matrix = create_interaction_matrix(df, num_users, num_items)
    recommendation_df = recommend_all_users(model, interaction_matrix, user_mapping, item_mapping, top_k=args.model_args.top_k, device=device)

    # Save Recommendations
    output_dir = os.path.dirname(args.dataset.output_path)
    os.makedirs(output_dir, exist_ok=True)
    recommendation_df.to_csv((args.dataset.output_path+"NCF.csv"), index=False)
    print(f"Recommendations saved to {args.dataset.output_path+"NCF.csv"}")