import numpy as np
import torch
from tqdm import tqdm

def recommend_top_k(model, train_data, device, k=10, batch_size=500):
    """
    훈련된 모델을 기반으로 각 사용자에게 top-k 추천 아이템을 생성합니다.
    
    Args:
        model (torch.nn.Module): 훈련된 VAE 모델.
        train_data (csr_matrix): 사용자-아이템 상호작용 행렬 (CSR 형식).
        device (torch.device): 연산을 수행할 장치 (CPU 또는 GPU).
        k (int): 추천할 아이템의 개수 (기본값은 10).
        batch_size (int): 사용자 데이터를 배치 단위로 처리할 크기 (기본값은 500).
    
    Returns:
        recommendations (list of list): 각 사용자에 대해 추천된 top-k 아이템 목록.
        - 예시: [[아이템1, 아이템2, ..., 아이템k], [아이템1, 아이템2, ..., 아이템k], ...]
    """
    model.eval()  # 모델을 평가 모드로 설정
    num_users = train_data.shape[0]  # 전체 사용자 수
    recommendations = []  # 추천 결과를 저장할 리스트

    # 사용자 데이터를 배치 단위로 처리
    for start_idx in tqdm(range(0, num_users, batch_size), desc="Generating Recommendations"):
        end_idx = min(start_idx + batch_size, num_users)
        batch_users = np.arange(start_idx, end_idx)  # 현재 배치의 사용자 인덱스

        # 현재 배치의 사용자에 대한 상호작용 행렬을 가져옴
        ratings_in = torch.Tensor(train_data[batch_users].toarray()).to(device)

        with torch.no_grad():  # 파라미터 업데이트를 하지 않기 위해서
            # 모델을 통해 예측된 평점 (손실 계산은 하지 않음)
            ratings_pred = model(ratings_in, calculate_loss=False).cpu().numpy()

        # 기존에 상호작용했던 아이템은 추천 목록에서 제외
        ratings_pred[train_data[batch_users].nonzero()] = -np.inf

        # 각 사용자의 top-k 아이템을 추천
        top_k_items = np.argpartition(-ratings_pred, k, axis=1)[:, :k]
        # 추천된 아이템들을 내림차순으로 정렬
        top_k_sorted = np.argsort(-ratings_pred[np.arange(ratings_pred.shape[0])[:, None], top_k_items], axis=1)
        top_k_items_sorted = top_k_items[np.arange(ratings_pred.shape[0])[:, None], top_k_sorted]

        recommendations.extend(top_k_items_sorted.tolist())  # 결과에 추가

    return recommendations
