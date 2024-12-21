import pandas as pd
import numpy as np
import torch

def create_interaction_matrix(df, num_users, num_items):
    """
    주어진 데이터프레임을 바탕으로 사용자-아이템 상호작용 행렬을 생성하는 함수.

    Args:
        df (DataFrame): 사용자-아이템 상호작용 데이터를 포함하는 pandas DataFrame.
                        이 데이터프레임은 'user_id', 'item_id', 'interaction' 열을 가져야 합니다.
        num_users (int): 총 사용자 수.
        num_items (int): 총 아이템 수.

    Returns:
        scipy.sparse.lil_matrix: 사용자-아이템 상호작용 행렬 (희소 행렬 형식).
    """
    import scipy.sparse as sp
    interaction_matrix = sp.lil_matrix((num_users, num_items))
    
    # 사용자-아이템 상호작용 정보를 바탕으로 행렬에 값 할당
    for _, row in df.iterrows():
        interaction_matrix[row['user_id'], row['item_id']] = 1  # 상호작용이 있으면 1로 설정
    
    return interaction_matrix


class InteractionDataset(torch.utils.data.Dataset):
    """
    PyTorch의 Dataset을 상속하여, 사용자-아이템 상호작용 데이터를 처리하는 클래스.

    Args:
        df (DataFrame): 사용자-아이템 상호작용 데이터를 포함하는 pandas DataFrame.
                        이 데이터프레임은 'user_id', 'item_id', 'interaction' 열을 가져야 합니다.
    """
    
    def __init__(self, df):
        """
        클래스 초기화 함수로, 데이터프레임에서 사용자, 아이템, 상호작용 정보를 추출하여 저장.
        
        Args:
            df (DataFrame): 'user_id', 'item_id', 'interaction' 열을 포함하는 데이터프레임.
        """
        self.users = df['user_id'].values  # 사용자 ID 리스트
        self.items = df['item_id'].values  # 아이템 ID 리스트
        self.labels = df['interaction'].values  # 상호작용 정보 (1 또는 0)

    def __len__(self):
        """
        데이터셋의 길이를 반환하는 함수.

        Returns:
            int: 데이터셋의 크기 (행의 수).
        """
        return len(self.users)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 사용자, 아이템, 상호작용 데이터를 반환하는 함수.

        Args:
            idx (int): 데이터를 추출할 인덱스.

        Returns:
            tuple: (user_id, item_id, interaction)
        """
        return self.users[idx], self.items[idx], self.labels[idx]
