import numpy as np
from tqdm import tqdm

def generate_recommendations(model, user_item_matrix, num_users, K=10):
    """
    주어진 모델을 사용해 사용자별 추천 아이템을 생성하는 함수.

    Args:
        model (object): 학습된 추천 모델 객체. 예측 메서드(`predict`)를 구현해야 합니다.
        user_item_matrix (scipy.sparse matrix): 사용자-아이템 상호작용을 나타내는 희소 행렬.
                                            각 행은 사용자를, 각 열은 아이템을 나타냅니다.
        num_users (int): 전체 사용자 수.
        K (int, optional): 각 사용자에게 추천할 아이템 수. 기본값은 10.

    Returns:
        dict: 사용자 ID를 키로 하고, 추천 아이템 ID 목록을 값으로 가지는 딕셔너리.
    """
    user_recommendations = {}

    for user_id in tqdm(range(num_users)):
        user_vector = user_item_matrix[user_id]
        scores = model.predict(user_vector)
        scores = scores.ravel()
        user_interacted_items = user_vector.indices
        scores[user_interacted_items] = -np.inf
        top_items = np.argpartition(scores, -K)[-K:]
        top_items = top_items[np.argsort(-scores[top_items])]
        user_recommendations[user_id] = top_items

    return user_recommendations
