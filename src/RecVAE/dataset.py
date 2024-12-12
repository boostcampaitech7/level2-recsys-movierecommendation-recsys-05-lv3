import numpy as np
from scipy import sparse
import pandas as pd
import os

class DatasetLoader:
    def __init__(self, dataset_path, global_indexing=False):
        self.dataset_path = dataset_path
        self.global_indexing = global_indexing
        self.unique_sid = []
        self.unique_uid = []
        self.n_items = 0
        self.n_users = 0
        self.train_data = None

        # 데이터 로드
        self._load_unique_ids()
        self.train_data = self._load_train_data()

    def _load_unique_ids(self):
        """ unique_sid와 unique_uid 로드 """
        # unique_sid 로드
        with open(os.path.join(self.dataset_path, 'unique_sid.txt'), 'r') as f:
            self.unique_sid = [line.strip() for line in f]

        # unique_uid 로드
        with open(os.path.join(self.dataset_path, 'unique_uid.txt'), 'r') as f:
            self.unique_uid = [line.strip() for line in f]

        # 아이템과 사용자 수
        self.n_items = len(self.unique_sid)
        self.n_users = len(self.unique_uid)

    def _load_train_data(self):
        """ 학습 데이터를 sparse 행렬로 로드 """
        csv_file = os.path.join(self.dataset_path, 'train.csv')
        tp = pd.read_csv(csv_file)
        
        # global_indexing에 따라 사용자 수 설정
        n_users = self.n_users if self.global_indexing else tp['uid'].max() + 1

        # 사용자, 아이템 id와 데이터를 sparse matrix로 변환
        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data

    def get_data(self):
        """ 데이터를 float32로 반환 """
        data = self.train_data.astype('float32')
        return data, self.unique_sid, self.unique_uid
