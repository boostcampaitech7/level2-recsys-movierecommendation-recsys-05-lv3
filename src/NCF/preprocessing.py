import pandas as pd
import numpy as np
import torch

def create_interaction_matrix(df, num_users, num_items):
    """
    주어진 사용자-아이템 상호작용 데이터를 바탕으로 희소 행렬을 생성합니다.
    
    Args:
        df (pd.DataFrame): 사용자-아이템 상호작용 데이터를 포함하는 데이터프레임. 
                            컬럼은 'user_id', 'item_id' 형식이어야 함.
        num_users (int): 사용자 수.
        num_items (int): 아이템 수.
    
    Returns:
        scipy.sparse.lil_matrix: 희소 행렬 (LIL 형식).
    """
    import scipy.sparse as sp
    # 희소 행렬 초기화 (num_users x num_items 크기)
    interaction_matrix = sp.lil_matrix((num_users, num_items))
    
    # 데이터프레임을 기반으로 상호작용을 기록
    for _, row in df.iterrows():
        interaction_matrix[row['user_id'], row['item_id']] = 1
    
    return interaction_matrix

class InteractionDataset(torch.utils.data.Dataset):
    """
    PyTorch의 Dataset 클래스를 상속하여 사용자-아이템 상호작용 데이터를 처리하는 데이터셋 클래스입니다.
    
    Args:
        df (pd.DataFrame): 사용자-아이템 상호작용 데이터프레임. 
                            컬럼은 'user_id', 'item_id', 'interaction' 형식이어야 함.
    """
    def __init__(self, df):
        self.users = df['user_id'].values
        self.items = df['item_id'].values
        self.labels = df['interaction'].values

    def __len__(self):
        """
        데이터셋의 크기 (즉, 상호작용 수)를 반환합니다.
        
        Returns:
            int: 데이터셋의 크기.
        """
        return len(self.users)

    def __getitem__(self, idx):
        """
        특정 인덱스에 해당하는 사용자, 아이템, 상호작용 레이블을 반환합니다.
        
        Args:
            idx (int): 데이터셋의 인덱스.
        
        Returns:
            tuple: (user_id, item_id, interaction_label)
        """
        return self.users[idx], self.items[idx], self.labels[idx]

def preprocess_data(df):
    """
    사용자 및 아이템의 고유 ID를 매핑하여 숫자로 변환하고, 상호작용 값을 1로 설정하여 
    데이터프레임을 전처리합니다.
    
    Args:
        df (pd.DataFrame): 'user'와 'item' 컬럼을 가진 원본 데이터프레임.
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임 (user_id, item_id, interaction 컬럼 추가).
        dict: 사용자 ID와 인덱스 간의 매핑 딕셔너리.
        dict: 아이템 ID와 인덱스 간의 매핑 딕셔너리.
    """
    # 사용자와 아이템의 고유 ID를 인덱스로 변환
    user_mapping = {id: idx for idx, id in enumerate(df['user'].unique())}
    item_mapping = {id: idx for idx, id in enumerate(df['item'].unique())}
    
    # 사용자 및 아이템 ID 매핑 적용
    df['user_id'] = df['user'].map(user_mapping)
    df['item_id'] = df['item'].map(item_mapping)
    
    # 상호작용 값은 1로 설정
    df['interaction'] = 1
    
    return df, user_mapping, item_mapping

def negative_sampling(df, num_items, num_negatives=4):
    """
    부정 샘플링을 통해 사용자가 상호작용하지 않은 아이템을 샘플링하여 
    학습 데이터를 생성합니다.
    
    Args:
        df (pd.DataFrame): 사용자-아이템 상호작용 데이터프레임.
        num_items (int): 전체 아이템 수.
        num_negatives (int, optional): 각 사용자에 대해 샘플링할 부정 샘플의 수 (기본값: 4).
    
    Returns:
        pd.DataFrame: 부정 샘플링을 통해 생성된 새로운 데이터프레임 (user_id, item_id, interaction 컬럼 포함).
    """
    # 사용자별로 그룹화
    user_group = df.groupby('user_id')
    all_items = set(range(num_items))

    negative_samples = []
    
    # 각 사용자에 대해 부정 샘플링 수행
    for user, group in user_group:
        positive_items = set(group['item_id'])  # 사용자가 상호작용한 아이템들
        negative_candidates = list(all_items - positive_items)  # 상호작용하지 않은 아이템들
        
        # 부정 샘플의 수가 부족하면 가능한 만큼만 샘플링
        if len(negative_candidates) < num_negatives:
            num_negatives = len(negative_candidates)
        
        # 랜덤하게 부정 샘플 선택
        negative_items = np.random.choice(negative_candidates, size=num_negatives, replace=False)

        # 부정 샘플을 리스트에 추가
        for item in negative_items:
            negative_samples.append([user, item, 0])  # interaction 값은 0으로 설정 (부정 샘플)
    
    # 부정 샘플 데이터프레임 생성
    negative_df = pd.DataFrame(negative_samples, columns=['user_id', 'item_id', 'interaction'])
    
    return negative_df
