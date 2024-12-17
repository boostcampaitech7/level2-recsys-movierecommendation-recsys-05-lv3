import torch
from tqdm import tqdm

def recommend_top_k(model, final_data, num_users, k=10, device='cuda'):
    """
    주어진 모델을 사용하여 각 사용자에 대해 상위 k개의 추천 항목을 반환합니다.

    Args:
        model: 추천 모델 객체.
        final_data: 사용자-아이템 상호작용 데이터프레임.
        num_users: 총 사용자 수.
        k: 추천할 항목의 수.
        device: 모델을 실행할 장치 ('cuda' 또는 'cpu').

    Returns:
        dict: 사용자 ID를 키로 하고 추천된 아이템 ID 목록을 값으로 가지는 딕셔너리.
    """
    model.eval()
    model.to(device)
    all_recommendations = {}
    all_items = set(final_data['item_id'].unique()) 

    with tqdm(total=num_users, desc="Recommending", unit="user") as pbar:
        for user_id in range(num_users):
            interacted_items = final_data[(final_data['user_id'] == user_id) & (final_data['watched'] == 1)]['item_id'].tolist()
            
            candidates = list(all_items - set(interacted_items))

            user_input = torch.tensor([user_id], dtype=torch.long, device=device)
            user_embedded = model.user_embedding(user_input).squeeze(0)  

            candidate_input = torch.tensor(candidates, dtype=torch.long, device=device)
            candidate_embedded = model.item_embedding(candidate_input) 
            
            with torch.no_grad():
                scores = torch.matmul(candidate_embedded, user_embedded)  

            top_k_indices = torch.topk(scores, k=min(k, len(candidates))).indices.cpu().numpy()
            top_k_items = [candidates[idx] for idx in top_k_indices]

            all_recommendations[user_id] = top_k_items[:k]
            pbar.update(1)

    return all_recommendations