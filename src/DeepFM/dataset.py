import numpy as np
import torch
from torch.utils.data import Dataset


class PairwiseRecommendationDataset(Dataset):
    """
    사용자-아이템 쌍을 위한 데이터셋 클래스.

    Attributes:
        user_col: 사용자 ID 텐서.
        item_col: 아이템 ID 텐서.
        user_item_dict: 사용자별 시청한 아이템 딕셔너리.
        all_items: 전체 아이템 리스트.
        item_probs: 아이템 샘플링 확률.
        num_negatives: 부정 샘플 수.
        user_item_embeddings: 사용자-아이템 임베딩.
    """
    def __init__(self, user_col, item_col, user_item_dict, all_items, item_probs, user_item_embeddings, num_negatives=1):
        self.user_col = user_col.cpu()
        self.item_col = item_col.cpu()
        self.user_item_dict = user_item_dict
        self.all_items = np.array(all_items)
        self.item_probs = item_probs
        self.num_negatives = num_negatives
        self.user_item_embeddings = user_item_embeddings  # 사용자-아이템 임베딩

    def __len__(self):
        return len(self.user_col)

    def __getitem__(self, idx):
        user = self.user_col[idx]
        pos_item = self.item_col[idx]

        # Negative 샘플링: 사용자-아이템 임베딩 유사도 기반
        neg_items = self.sample_similarity_negatives(user)

        return user, pos_item, torch.tensor(neg_items, dtype=torch.long)

    def sample_similarity_negatives(self, user):
        """
        사용자-아이템 임베딩 유사도 기반으로 부정 샘플을 생성합니다.

        Args:
            user: 사용자 ID.

        Returns:
            list: 부정 아이템 ID 리스트.
        """
        user_id = user.item()
        seen_items = self.user_item_dict[user_id]

        # 유저 임베딩
        user_embedding = self.user_item_embeddings[user_id]
        # 모든 아이템 임베딩과의 유사도 계산 (Cosine Similarity)
        item_embeddings = self.user_item_embeddings[len(self.user_item_dict):]
        similarities = np.dot(item_embeddings, user_embedding) / (
            np.linalg.norm(item_embeddings, axis=1) * np.linalg.norm(user_embedding)
        )

        # 유사도가 낮은 아이템 중 Negative 샘플 선택
        neg_items = []
        while len(neg_items) < self.num_negatives:
            neg_item = np.random.choice(self.all_items, p=self.item_probs)
            if neg_item not in seen_items:
                neg_items.append(neg_item)
        return neg_items