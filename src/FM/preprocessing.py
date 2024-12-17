import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

class TextDataset(Dataset):
    """
    PyTorch Dataset 클래스. 영화 추천 모델을 위한 데이터셋 클래스입니다.
    
    Attributes:
        feature_vectors (numpy.ndarray): 모델 학습에 사용되는 특징 벡터.
        rating (numpy.ndarray, optional): 각 샘플의 평점. 디폴트 값은 None.
    """
    def __init__(self, feature_vectors, rating=None):
        """
        Args:
            feature_vectors (numpy.ndarray): 모델에 입력될 특징 벡터.
            rating (numpy.ndarray, optional): 샘플에 대한 평점. 디폴트는 None.
        """
        self.feature_vectors = np.stack(feature_vectors)
        self.rating = rating

    def __len__(self):
        """ 데이터셋의 크기 반환 """
        return len(self.feature_vectors)
    
    def __getitem__(self, i):
        """ 인덱스를 통해 데이터 샘플을 반환 """
        data = {
            'features': torch.tensor(self.feature_vectors[i], dtype=torch.float32),
            'rating': torch.tensor(self.rating[i], dtype=torch.float32)
        }
        return data


def load_rating_data(file_path):
    """
    평점 데이터를 로드하고, 모든 평점을 암시적 피드백(1.0)으로 설정하는 함수.
    
    Args:
        file_path (str): 평점 데이터가 저장된 CSV 파일 경로.
        
    Returns:
        pandas.DataFrame: 로드된 평점 데이터프레임.
    """
    raw_rating_df = pd.read_csv(file_path)
    raw_rating_df['rating'] = 1.0  # implicit feedback
    raw_rating_df.drop(['time'], axis=1, inplace=True)
    return raw_rating_df


def load_director_data(file_path, embedding_dim=10):
    """
    감독 데이터를 로드하고, 각 감독에 대해 임베딩을 생성하는 함수.
    
    Args:
        file_path (str): 감독 데이터가 저장된 파일 경로.
        embedding_dim (int, optional): 감독 임베딩의 차원. 기본값은 10.
        
    Returns:
        pandas.DataFrame: 아이템과 감독 벡터가 포함된 데이터프레임.
    """
    raw_director_df = pd.read_csv(file_path, sep='\t')
    director_encoder = LabelEncoder()
    raw_director_df['encoded_director'] = director_encoder.fit_transform(raw_director_df['director'])
    
    director_embedding = nn.Embedding(raw_director_df['encoded_director'].nunique(), embedding_dim)
    director_grouped = raw_director_df.groupby('item')['encoded_director'].apply(list).sort_index()
    
    director_avg_vectors = []
    for ids in director_grouped:
        director_embeds = director_embedding(torch.tensor(ids))
        avg_vector = torch.mean(director_embeds, dim=0)
        director_avg_vectors.append(avg_vector.detach().numpy())
    
    director_ = np.concatenate([director_grouped.index.to_numpy().reshape(-1, 1), director_avg_vectors], axis=1)
    return pd.DataFrame({'item': director_[:, 0], 'director_vectors': list(director_[:, 1:].astype(np.float32))})


def load_writer_data(file_path, embedding_dim=10):
    """
    작가 데이터를 로드하고, 각 작가에 대해 임베딩을 생성하는 함수.
    
    Args:
        file_path (str): 작가 데이터가 저장된 파일 경로.
        embedding_dim (int, optional): 작가 임베딩의 차원. 기본값은 10.
        
    Returns:
        pandas.DataFrame: 아이템과 작가 벡터가 포함된 데이터프레임.
    """
    raw_writer_df = pd.read_csv(file_path, sep='\t')
    writer_encoder = LabelEncoder()
    raw_writer_df['encoded_writer'] = writer_encoder.fit_transform(raw_writer_df['writer'])
    
    writer_embedding = nn.Embedding(raw_writer_df['encoded_writer'].nunique(), embedding_dim)
    writer_grouped = raw_writer_df.groupby('item')['encoded_writer'].apply(list).sort_index()
    
    writer_avg_vectors = []
    for ids in writer_grouped:
        writer_embeds = writer_embedding(torch.tensor(ids))
        avg_vector = torch.mean(writer_embeds, dim=0)
        writer_avg_vectors.append(avg_vector.detach().numpy())
    
    writer_ = np.concatenate([writer_grouped.index.to_numpy().reshape(-1, 1), writer_avg_vectors], axis=1)
    return pd.DataFrame({'item': writer_[:, 0], 'writer_vectors': list(writer_[:, 1:].astype(np.float32))})


def load_year_data(file_path):
    """
    연도 데이터를 로드하는 함수.
    
    Args:
        file_path (str): 연도 데이터가 저장된 파일 경로.
        
    Returns:
        pandas.DataFrame: 연도 데이터프레임.
    """
    return pd.read_csv(file_path, sep='\t')


def load_genre_data(file_path):
    """
    장르 데이터를 로드하고, 다중 레이블을 원-핫 인코딩하는 함수.
    
    Args:
        file_path (str): 장르 데이터가 저장된 파일 경로.
        
    Returns:
        pandas.DataFrame: 아이템과 장르 벡터가 포함된 데이터프레임.
    """
    raw_genre_df = pd.read_csv(file_path, sep='\t')
    
    genre_multi_hot = raw_genre_df.groupby('item')['genre'].apply(
        lambda x: np.array([1 if genre in x.tolist() else 0 for genre in raw_genre_df['genre'].unique()]))
    
    genre_ = np.concatenate([genre_multi_hot.index.to_numpy().reshape(-1, 1), np.array(genre_multi_hot).reshape(-1, 1)], axis=1)
    
    return pd.DataFrame(genre_, columns=['item', 'genre_vectors'])


def negative_sampling(raw_rating_df, users):
    """
    부정 샘플을 생성하여 데이터셋에 추가하는 함수. 
    각 사용자에 대해 시청하지 않은 아이템을 랜덤하게 샘플링하여 부정 샘플을 만듦.
    
    Args:
        raw_rating_df (pandas.DataFrame): 원본 평점 데이터프레임.
        users (array-like): 사용자 리스트.
        
    Returns:
        pandas.DataFrame: 부정 샘플이 추가된 평점 데이터프레임.
    """
    negative_samples = []
    for user in tqdm(users):
        # 시청한 영화
        positive_items = set(raw_rating_df[raw_rating_df['user'] == user]['item'])
        # 시청하지 않은 영화
        negative_items = set(raw_rating_df['item'].unique()) - positive_items
        num_negatives = 50
        sampled_negatives = np.random.choice(list(negative_items), num_negatives, replace=False)
        for neg_movie in sampled_negatives:
            negative_samples.append({'user': user, 'item': neg_movie, 'rating': 0})

    negative_df = pd.DataFrame(negative_samples)
    rating_df = pd.concat([raw_rating_df, negative_df], ignore_index=True)

    return rating_df


def fill_na_vectors(merged_df):
    """
    결측값이 있는 벡터를 0으로 채우는 함수.
    
    Args:
        merged_df (pandas.DataFrame): 결합된 데이터프레임.
        
    Returns:
        pandas.DataFrame: 결측값이 처리된 데이터프레임.
    """
    merged_df['director_vectors'] = merged_df['director_vectors'].apply(
        lambda x: np.zeros(10) if not isinstance(x, np.ndarray) else x)
    merged_df['writer_vectors'] = merged_df['writer_vectors'].apply(
        lambda x: np.zeros(10) if not isinstance(x, np.ndarray) else x)
    merged_df['genre_vectors'] = merged_df['genre_vectors'].apply(
        lambda x: np.zeros(18) if not isinstance(x, np.ndarray) else x)
    merged_df['year'] = merged_df['year'].fillna(0).astype(np.float32)
    return merged_df


def create_feature_vectors_for_train(merged_df):
    """
    학습을 위한 특징 벡터를 생성하는 함수. 각 샘플에 대해 특징 벡터를 결합하여 생성.
    
    Args:
        merged_df (pandas.DataFrame): 아이템과 관련된 모든 데이터를 포함하는 데이터프레임.
        
    Returns:
        pandas.DataFrame: 생성된 특징 벡터를 포함한 데이터프레임.
    """
    feature_vectors = []
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="create_feature_vectors_for_train"):
        feature_vector = np.concatenate([
            np.array([row['user']]),               # user ID
            np.array([row['item']]),               # item ID
            row['director_vectors'],               # 감독 벡터
            row['writer_vectors'],                 # 작가 벡터
            np.array([row['year']]),               # 연도 정보
            row['genre_vectors']                   # 장르 벡터
        ])
        feature_vectors.append(feature_vector)
    
    merged_df['feature_vectors'] = feature_vectors
    return merged_df

def create_feature_vectors_for_inference(merged_df):
    """
    모델 추론을 위한 특징 벡터를 생성하는 함수. 이 함수는 사용자 정보 없이 각 아이템에 대한 특징 벡터를 생성합니다.

    Args:
        merged_df (pandas.DataFrame): 아이템에 대한 감독, 작가, 연도, 장르 정보가 포함된 데이터프레임.

    Returns:
        torch.Tensor: 각 아이템에 대한 특징 벡터를 담고 있는 텐서.
    """
    # 추론에 사용할 아이템별 데이터 선택
    for_inference_df = merged_df[['item','director_vectors', 'writer_vectors', 'year', 'genre_vectors']]
    for_inference_df.drop_duplicates(subset=['item'], keep='first', inplace=True)

    # 특징 벡터 생성
    no_user_feature_vectors = []
    for _, row in tqdm(for_inference_df.iterrows(), total=len(for_inference_df), desc="create_feature_vectors_for_inference"):
        no_user_feature_vector = np.concatenate([
            np.array([row['item']]),         # item ID
            row['director_vectors'],         # 감독 벡터
            row['writer_vectors'],           # 작가 벡터
            np.array([row['year']]),         # 연도 정보
            row['genre_vectors']             # 장르 벡터
        ])
        no_user_feature_vectors.append(no_user_feature_vector)
    
    # 텐서 형태로 변환하여 반환
    no_user_feature_vectors = torch.tensor(no_user_feature_vectors)
    return no_user_feature_vectors


def data_processing(data):
    """
    데이터에서 사용자와 아이템을 라벨 인코딩하여 모델 학습에 필요한 데이터로 변환하는 함수.

    Args:
        data (pandas.DataFrame): 원본 데이터프레임, 'user'와 'item' 열이 포함되어 있어야 합니다.

    Returns:
        dict: 라벨 인코딩된 데이터와 추가적인 메타데이터를 포함한 딕셔너리.
    """
    sparse_cols = ['user', 'item']

    # 각 컬럼에 대해 라벨 인코딩 수행
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        data[col] = data[col].fillna('unknown')  # 결측값 처리
        unique_labels = data[col].astype("category").cat.categories
        label2idx[col] = {label: idx for idx, label in enumerate(unique_labels)}  # 라벨 -> 인덱스
        idx2label[col] = {idx: label for idx, label in enumerate(unique_labels)}  # 인덱스 -> 라벨
        data[col] = data[col].map(label2idx[col])  # 라벨을 인덱스로 변환

    # 필드 크기 정보 저장
    field_dims = [len(label2idx[col]) for col in sparse_cols]

    return {
        'train': data,
        'field_names': sparse_cols,
        'field_dims': field_dims,
        'label2idx': label2idx,
        'idx2label': idx2label,
    }


def data_split(data, valid_ratio=0.1):
    """
    주어진 데이터를 훈련 세트와 검증 세트로 분할하는 함수.

    Args:
        data (dict): 라벨 인코딩된 데이터를 포함한 딕셔너리.
        valid_ratio (float, optional): 검증 데이터의 비율. 기본값은 0.1.

    Returns:
        dict: 훈련 및 검증 데이터로 분할된 데이터 딕셔너리.
    """
    if valid_ratio == 0:
        data['X_train'] = data['train'].drop('rating', axis=1)
        data['y_train'] = data['train']['rating']
    else:
        # 훈련 데이터와 검증 데이터 분할
        X_train, X_valid, y_train, y_valid = train_test_split(
            data['train'].drop(['rating'], axis=1),
            data['train']['rating'],
            test_size=valid_ratio,
            shuffle=True
        )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    return data


def data_loader(data, batch_size, shuffle, valid_ratio=0.1):
    """
    훈련과 검증 데이터를 PyTorch DataLoader 형식으로 반환하는 함수.
    
    Args:
        data (dict): 훈련 및 검증 데이터를 포함한 딕셔너리.
        batch_size (int): 배치 크기.
        shuffle (bool): 훈련 데이터를 섞을지 여부.
        valid_ratio (float, optional): 검증 데이터 비율. 기본값은 0.1.
    
    Returns:
        dict: 훈련 및 검증용 DataLoader가 포함된 딕셔너리.
    """
    # 훈련 데이터셋 생성
    train_features = data['X_train']['feature_vectors']
    train_ratings = data['y_train'].values
    train_dataset = TextDataset(train_features, train_ratings)

    valid_dataset = None
    # 검증 데이터셋이 있을 경우 생성
    if valid_ratio != 0:
        valid_features = data['X_valid']['feature_vectors']
        valid_ratings = data['y_valid'].values
        valid_dataset = TextDataset(valid_features, valid_ratings)

    # DataLoader 생성
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) if valid_ratio else None

    # DataLoader 딕셔너리에 저장
    data['train_dataloader'], data['valid_dataloader'] = train_dataloader, valid_dataloader
    
    return data
