import pandas as pd
import numpy as np
import torch


def inference(model, feature, device):
    """
    주어진 특징(feature)에 대해 모델을 사용하여 예측을 수행하는 함수입니다.

    Args:
        model (torch.nn.Module): 예측을 수행할 모델.
        feature (numpy.ndarray): 모델에 입력될 특징 데이터. 2D 배열이어야 하며, 각 행은 하나의 샘플을 나타냅니다.
        device (torch.device): 모델을 실행할 디바이스(CPU 또는 GPU).

    Returns:
        numpy.ndarray: 예측된 확률값. sigmoid 함수를 통과한 후의 확률값이 반환됩니다.
    """
    model.eval()

    # feature가 numpy 배열이고 object 타입일 경우, float32로 변환
    if isinstance(feature, np.ndarray) and feature.dtype == object:
        feature = np.vstack(feature).astype(np.float32)
    
    with torch.no_grad():
        x = torch.tensor(feature, dtype=torch.float).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

def generate_recommendation(model, raw_rating_df, feature_vector, device, k=10):
    """
    주어진 사용자-아이템 평점 데이터와 특징 벡터를 바탕으로 추천을 생성하는 함수입니다.

    Args:
        model (torch.nn.Module): 추천을 위한 예측 모델.
        raw_rating_df (pandas.DataFrame): 사용자-아이템 평점 데이터프레임. 'user'와 'item' 컬럼이 포함되어 있어야 합니다.
        feature_vector (numpy.ndarray): 각 아이템에 대한 특징 벡터.
        device (torch.device): 모델을 실행할 디바이스(CPU 또는 GPU).
        k (int, optional): 각 사용자에게 추천할 아이템의 수. 기본값은 10.

    Returns:
        pandas.DataFrame: 각 사용자에게 추천된 아이템들의 정보가 담긴 데이터프레임. 'user'와 'item' 컬럼이 포함됩니다.
    """
    users = raw_rating_df['user'].unique()
    items = raw_rating_df['item'].unique()

    rusers_dict = {i: users[i] for i in range(len(users))}
    ritems_dict = {i: items[i] for i in range(len(items))}

    model.eval()
    recommendations = []

    with torch.no_grad():
        # 각 사용자에 대해 추천 생성
        for user in range(len(users)):
            user_ = torch.full((len(items),), user, dtype=torch.float)
            x = torch.cat([user_.unsqueeze(1), feature_vector], dim=1)
            score = inference(model, x, device)
            top_k_indices = np.argsort(score)[-k:]

            # 추천 결과 저장
            for top_k_index in top_k_indices:
                recommendations.append({'user': rusers_dict[user], 'item': ritems_dict[top_k_index]})
    
    recommendation_df = pd.DataFrame(recommendations)
    return recommendation_df
