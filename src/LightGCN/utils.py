import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    """
    학습용 데이터셋 클래스

    이 클래스는 PyTorch의 Dataset을 상속받아 LightGCN 모델 학습을 위한 데이터셋을 생성합니다.

    매개변수:
        df (pandas.DataFrame): 사용자-아이템 상호작용 데이터프레임
        num_items (int): 전체 아이템 수

    속성:
        users (numpy.array): 사용자 ID 배열
        pos_items (numpy.array): 긍정적 아이템 ID 배열
        num_items (int): 전체 아이템 수
    """
    def __init__(self, df, num_items):

        self.users = df['user'].values
        self.pos_items = df['item'].values
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        neg_item = np.random.randint(self.num_items)
        while neg_item in self.pos_items[self.users == user]:
            neg_item = np.random.randint(self.num_items)
        return user, pos_item, neg_item

def create_adj_matrix(df, num_users, num_items):
    """
    정규화된 인접 행렬 생성 함수

    사용자-아이템 상호작용 데이터를 기반으로 정규화된 인접 행렬을 생성합니다.

    매개변수:
        df (pandas.DataFrame): 사용자-아이템 상호작용 데이터프레임
        num_users (int): 전체 사용자 수
        num_items (int): 전체 아이템 수

    반환값:
        torch.sparse.FloatTensor: 정규화된 인접 행렬의 희소 텐서 표현
    """
    user_nodes = df['user'].values
    item_nodes = df['item'].values + num_users
    edge_index = np.array([np.concatenate([user_nodes, item_nodes]),
                          np.concatenate([item_nodes, user_nodes])])
    adj = sp.coo_matrix((np.ones(len(edge_index[0])), 
                        (edge_index[0], edge_index[1])),
                       shape=(num_users + num_items, num_users + num_items),
                       dtype=np.float32)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    return torch.sparse.FloatTensor(
        torch.LongTensor(np.array([norm_adj.row, norm_adj.col])),
        torch.FloatTensor(norm_adj.data),
        torch.Size(norm_adj.shape)
    )

def create_submission(recommendations, user_id_map, item_id_map, output_path):
    """
    추천 결과를 제출 파일로 생성하는 함수

    모델의 추천 결과를 CSV 형식의 제출 파일로 변환합니다.

    매개변수:
        recommendations (dict): 사용자별 추천 아이템 딕셔너리
        user_id_map (pandas.DataFrame): 사용자 ID 매핑 정보
        item_id_map (pandas.DataFrame): 아이템 ID 매핑 정보
        output_path (str): 제출 파일 저장 경로

    반환값:
        None
    """
    users = []
    items = []
    
    original_user_ids = user_id_map['original_id'].tolist()
    original_item_ids = item_id_map['original_id'].tolist()
    
    for user, rec_items in recommendations.items():
        users.extend([original_user_ids[user]] * 10)
        items.extend([original_item_ids[item] for item in rec_items])
    
    submission_df = pd.DataFrame({
        'user': users,
        'item': items
    })
    
    submission_df.to_csv(output_path, index=False)
