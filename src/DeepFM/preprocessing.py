import os
import pandas as pd
import numpy as np
import re
from gensim.models import FastText
from sklearn.preprocessing import LabelEncoder

def load_data(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
    year_data = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    title_data = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
    genre_data = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
    return train_df, year_data, title_data, genre_data

def preprocess_train_data(train_df):
    train_df['watch_date'] = pd.to_datetime(train_df['time'], unit='s')
    train_df.drop(columns=['time'], inplace=True)
    train_df['watch_year'] = train_df['watch_date'].dt.year
    train_df.drop(columns=['watch_date'], inplace=True)
    return train_df

def merge_data(train_df, title_data, year_data):
    all_data = pd.merge(train_df, title_data, on='item', how='left').merge(year_data, on='item', how='left')
    return all_data

def clean_titles(all_data):
    all_data.loc[all_data['item'] == 34048, 'title'] += " Extended"
    all_data['extracted_year'] = all_data['title'].apply(lambda x: int(re.search(r'\((\d{4})\)', x).group(1)) if re.search(r'\((\d{4})\)', x) else None)
    all_data['year'] = all_data['year'].fillna(all_data['extracted_year']).astype('int32')
    all_data.drop(columns=['extracted_year'], inplace=True)
    all_data['title'] = all_data['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)', '', x))
    return all_data

def tokenize_titles(all_data):
    all_data['title_tokens'] = all_data['title'].apply(lambda x: x.split())
    return all_data

def train_fasttext_model(all_data):
    fasttext_model = FastText(
        sentences=all_data['title_tokens'],
        vector_size=50,
        window=3,
        min_count=1,
        sg=1,
        epochs=10
    )
    return fasttext_model

def get_title_embedding(title_tokens, model, vector_size):
    vectors = [model.wv[word] for word in title_tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

def add_title_embeddings(all_data, model, embedding_dim):
    all_data['title_embedding'] = all_data['title_tokens'].apply(lambda x: get_title_embedding(x, model, embedding_dim))
    return all_data.drop(columns=['title_tokens', 'title'])

def process_genres(genre_data):
    genre_data['genre_list'] = genre_data['genre'].apply(lambda x: x.split(','))
    genres_per_item = genre_data.groupby('item')['genre_list'].agg(lambda x: sum(x, [])).reset_index()
    return genres_per_item

def merge_genres(all_data, genres_per_item):
    all_data = pd.merge(all_data, genres_per_item, on='item', how='left')
    return all_data

def encode_genres(all_genres, genre_list):
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
    
    def encode_genre_list(genre_list):
        multi_hot = np.zeros(len(all_genres), dtype=int)
        for genre in genre_list:
            if genre in genre_to_idx:
                multi_hot[genre_to_idx[genre]] = 1
        return multi_hot
    
    return encode_genre_list

def add_genre_embeddings(all_data):
    all_genres = sorted(set([genre for sublist in all_data['genre_list'] for genre in sublist]))
    
    encode_genre_list_fn = encode_genres(all_genres)
    
    all_data['genre_embedding'] = all_data['genre_list'].apply(encode_genre_list_fn)
    
    return all_genres, all_data.drop(columns=['genre_list'])

def label_encode_users_items(all_data):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    all_data['user_encoded'] = user_encoder.fit_transform(all_data['user'])
    all_data['item_encoded'] = item_encoder.fit_transform(all_data['item'])
    
    columns_ordered = ['user_encoded', 'item_encoded'] + [col for col in all_data.columns if col not in ['user_encoded', 'item_encoded']]
    
    user_id_mapping = dict(zip(all_data['user_encoded'], all_data['user']))
    item_id_mapping = dict(zip(all_data['item_encoded'], all_data['item']))
    
    return user_id_mapping, item_id_mapping, all_data[columns_ordered]

