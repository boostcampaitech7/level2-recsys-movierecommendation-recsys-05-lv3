import torch
from tqdm import tqdm

def recommend_top_k(model, final_data, num_users, k=10, device='cuda'):

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