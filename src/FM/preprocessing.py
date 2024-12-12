import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

class Text_Dataset(Dataset):
    def __init__(self, feature_vectors, rating=None):
        self.feature_vectors = np.stack(feature_vectors)
        self.rating = rating

    def __len__(self):
        return len(self.feature_vectors)
    
    def __getitem__(self, i):
        data = {
            'features': torch.tensor(self.feature_vectors[i], dtype=torch.float32),
            'rating': torch.tensor(self.rating[i], dtype=torch.float32)
        }
        return data


def load_rating_data(file_path):
    raw_rating_df = pd.read_csv(file_path)
    raw_rating_df['rating'] = 1.0  # implicit feedback
    raw_rating_df.drop(['time'], axis=1, inplace=True)
    return raw_rating_df


def load_director_data(file_path, embedding_dim=10):
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
    return pd.read_csv(file_path, sep='\t')


def load_genre_data(file_path):
    raw_genre_df = pd.read_csv(file_path, sep='\t')
    
    genre_multi_hot = raw_genre_df.groupby('item')['genre'].apply(
        lambda x: np.array([1 if genre in x.tolist() else 0 for genre in raw_genre_df['genre'].unique()]))
    
    genre_ = np.concatenate([genre_multi_hot.index.to_numpy().reshape(-1, 1), np.array(genre_multi_hot).reshape(-1, 1)], axis=1)
    
    return pd.DataFrame(genre_, columns=['item', 'genre_vectors'])


def negative_sampling(raw_rating_df, users):
    negative_samples = []
    for user in tqdm(users):
        users = raw_rating_df['user'].unique()
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
    merged_df['director_vectors'] = merged_df['director_vectors'].apply(
        lambda x: np.zeros(10) if not isinstance(x, np.ndarray) else x)
    merged_df['writer_vectors'] = merged_df['writer_vectors'].apply(
        lambda x: np.zeros(10) if not isinstance(x, np.ndarray) else x)
    merged_df['genre_vectors'] = merged_df['genre_vectors'].apply(
        lambda x: np.zeros(18) if not isinstance(x, np.ndarray) else x)
    merged_df['year'] = merged_df['year'].fillna(0).astype(np.float32)
    return merged_df


def create_feature_vectors_for_train(merged_df):
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
    for_inference_df = merged_df[['item','director_vectors', 'writer_vectors', 'year', 'genre_vectors']]
    for_inference_df.drop_duplicates(subset=['item'], keep='first', inplace=True)

    no_user_feature_vectors = []
    for _, row in tqdm(for_inference_df.iterrows(), total=len(for_inference_df), desc="create_feature_vectors_for_inference"):
        no_user_feature_vector = np.concatenate([
            np.array([row['item']]),         # item ID
            row['director_vectors'],               # 감독 벡터
            row['writer_vectors'],                 # 작가 벡터
            np.array([row['year']]),               # 연도 정보
            row['genre_vectors']                   # 장르 벡터
        ])
        #no_user_feature_vectors = torch.tensor(no_user_feature_vector)
        no_user_feature_vectors.append(no_user_feature_vector)
    no_user_feature_vectors = torch.tensor(no_user_feature_vectors)
    return no_user_feature_vectors


def data_processing(data):
    sparse_cols = ['user','item']

    # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        data[col] = data[col].fillna('unknown')
        unique_labels = data[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        data[col] = data[col].map(label2idx[col])

    field_dims = [len(label2idx[col]) for col in sparse_cols]

    data = {
            'train':data,
            'field_names':sparse_cols,
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            }
    
    return data


def data_split(data, valid_ratio=0.1):
    
    if valid_ratio == 0:
        data['X_train'] = data['train'].drop('rating', axis=1)
        data['y_train'] = data['train']['rating']

    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
                                                            data['train'].drop(['rating'], axis=1),
                                                            data['train']['rating'],
                                                            test_size=valid_ratio,
                                                            shuffle=True
                                                            )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    return data


def data_loader(data, batch_size, shuffle, valid_ratio=0.1):

    train_features = data['X_train']['feature_vectors']
    train_ratings = data['y_train'].values
    train_dataset = Text_Dataset(train_features, train_ratings)

    valid_dataset = None
    if valid_ratio != 0:
        valid_features = data['X_valid']['feature_vectors']
        valid_ratings = data['y_valid'].values
        valid_dataset = Text_Dataset(valid_features, valid_ratings)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) if valid_ratio else None

    data['train_dataloader'], data['valid_dataloader'] = train_dataloader, valid_dataloader
    
    return data
