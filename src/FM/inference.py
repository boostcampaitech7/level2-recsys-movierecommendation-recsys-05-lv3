import pandas as pd
import numpy as np
import torch


def inference(model, feature, device):
    model.eval()

    if isinstance(feature, np.ndarray) and feature.dtype == object:
        feature = np.vstack(feature).astype(np.float32)
    
    with torch.no_grad():
        x = torch.tensor(feature, dtype=torch.float).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

def generate_recommendation(model, raw_rating_df, feature_vector, device, k=10):
    users = raw_rating_df['user'].unique()
    items = raw_rating_df['item'].unique()

    rusers_dict = {i: users[i] for i in range(len(users))}
    ritems_dict = {i: items[i] for i in range(len(items))}

    model.eval()
    recommendations = []

    with torch.no_grad():
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
