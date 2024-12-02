import numpy as np
from tqdm import tqdm

def generate_recommendations(model, user_item_matrix, num_users, K=10):
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
