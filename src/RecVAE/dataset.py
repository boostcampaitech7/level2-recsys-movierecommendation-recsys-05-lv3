import numpy as np
from scipy import sparse
import pandas as pd
import os

class DatasetLoader:
    """
    데이터셋을 로드하는 클래스입니다. 이 클래스는 주어진 경로에서 사용자 및 아이템의 고유 ID를 읽고, 
    학습 데이터를 sparse 행렬 형태로 로드하여 모델 학습에 사용할 수 있도록 제공합니다.
    
    Attributes:
        dataset_path (str): 데이터셋이 저장된 디렉토리 경로.
        global_indexing (bool): 글로벌 인덱싱 여부 (True일 경우 전체 사용자 범위, False일 경우 최대 사용자 ID에 맞춤).
        unique_sid (list): 고유한 아이템 ID 목록.
        unique_uid (list): 고유한 사용자 ID 목록.
        n_items (int): 아이템의 개수.
        n_users (int): 사용자의 개수.
        train_data (csr_matrix): 학습 데이터 (sparse 행렬).
    """
    
    def __init__(self, dataset_path, global_indexing=False):
        """
        클래스 초기화 메서드입니다. 주어진 경로에서 데이터를 로드하고 초기화합니다.
        
        Parameters:
            dataset_path (str): 데이터셋이 저장된 디렉토리 경로.
            global_indexing (bool): 글로벌 인덱싱 여부 (True일 경우 전체 사용자 범위, False일 경우 최대 사용자 ID에 맞춤).
        """
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
        """
        고유한 아이템 ID(`unique_sid`)와 고유한 사용자 ID(`unique_uid`)를 로드합니다.
        - 'unique_sid.txt' 파일에서 아이템 ID 목록을 읽어옵니다.
        - 'unique_uid.txt' 파일에서 사용자 ID 목록을 읽어옵니다.
        """
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
        """
        학습 데이터를 sparse 행렬 형태로 로드합니다.
        - 'train.csv' 파일에서 사용자 ID(`uid`)와 아이템 ID(`sid`)를 읽고, sparse 행렬로 변환합니다.
        - 글로벌 인덱싱 여부(`global_indexing`)에 따라 사용자 수를 설정합니다.
        
        Returns:
            sparse.csr_matrix: 변환된 학습 데이터 sparse 행렬.
        """
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
        """
        로드한 학습 데이터를 float32 타입으로 반환합니다.
        
        Returns:
            tuple: 학습 데이터 (sparse matrix), 고유 아이템 ID 목록, 고유 사용자 ID 목록
        """
        # 학습 데이터를 float32로 변환하여 반환
        data = self.train_data.astype('float32')
        return data, self.unique_sid, self.unique_uid
