import torch
import numpy as np 

def recommend_top_k(model, user_item_dict, all_items, k=10):
    """
    모델을 사용하여 모든 유저에 대해 상위 K개의 영화를 추천합니다.
    
    Args:
    - model: 훈련된 DeepFM 모델.
    - user_item_dict: 유저별 시청한 영화 딕셔너리 {user_id: set(movies_seen)}.
    - all_items: 전체 아이템 리스트.
    - k: 추천할 영화 개수.
    
    Returns:
    - recommendations: 추천 결과 리스트 [{"user": user_id, "item": movie_id}, ...]
    """
    model.eval()
    recommendations = []
    
    with torch.no_grad():
        for user_id in user_item_dict.keys():
            # 유저가 본 적 없는 영화
            unseen_items = list(set(all_items) - user_item_dict[user_id])
            
            if len(unseen_items) == 0:
                continue  # 모든 영화를 이미 본 경우
            
            # 유저 ID와 미시청 영화 ID 텐서 생성
            user_tensor = torch.tensor([user_id] * len(unseen_items), dtype=torch.long, device=device)
            item_tensor = torch.tensor(unseen_items, dtype=torch.long, device=device)
            
            # Title과 Genre 임베딩
            genre_tensor = genre_col[item_tensor].to(device)
            
            # Year과 Watch Year
            year_tensor = torch.tensor(
                all_data['year'].values[item_tensor.cpu().numpy()],
                dtype=torch.float32,
                device=device
            ).unsqueeze(1)
            
            watch_year_tensor = torch.tensor(
                all_data['watch_year'].values[item_tensor.cpu().numpy()],
                dtype=torch.float32,
                device=device
            ).unsqueeze(1)
            
            # Continuous Features 결합
            continuous_features = torch.cat([genre_tensor, year_tensor, watch_year_tensor], dim=1)
            
            # Categorical Features 결합
            categorical_features = torch.stack([user_tensor, item_tensor], dim=1)
            
            # 모델로 점수 예측
            scores = model(categorical_features, continuous_features)
            
            # 상위 K개 아이템 선택
            top_k_indices = torch.topk(scores, k=k).indices.cpu().numpy()
            top_k_items = item_tensor[top_k_indices].cpu().numpy()
            
            # 추천 결과 저장
            for item_id in top_k_items:
                recommendations.append({"user": user_id, "item": item_id})
    
    return recommendations
