import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import LabelEncoder
from gensim.models import FastText

def load_data(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
    year_data = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    writer_data = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
    title_data = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
    genre_data = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
    director_data = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')
    return train_df, year_data, title_data, genre_data

def preprocess_data(train_df, title_data, year_data):
    train_df['watch_date'] = pd.to_datetime(train_df['time'], unit='s')
    train_df.drop(columns=['time'], inplace=True)
    all_data = pd.merge(train_df, title_data, on='item', how='left').merge(year_data, on='item', how='left')
    all_data['watch_year'] = all_data['watch_date'].dt.year
    all_data.drop(columns=['watch_date'], inplace=True)
    all_data.loc[all_data['item'] == 34048, 'title'] += " Extended"
    all_data['extracted_year'] = all_data['title'].apply(lambda x: int(re.search(r'\((\d{4})\)', x).group(1)) if re.search(r'\((\d{4})\)', x) else None)
    all_data['year'] = all_data['year'].fillna(all_data['extracted_year']).astype('int32')
    all_data.drop(columns=['extracted_year'], inplace=True)
    all_data['title'] = all_data['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)', '', x))
    return all_data

def train_fasttext(all_data):
    all_data['title_tokens'] = all_data['title'].apply(lambda x: x.split())
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
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

def encode_genres(all_data, genre_data):
    genre_data['genre_list'] = genre_data['genre'].apply(lambda x: x.split(','))
    genres_per_item = genre_data.groupby('item')['genre_list'].agg(lambda x: sum(x, [])).reset_index()
    all_data = pd.merge(all_data, genres_per_item, on='item', how='left')
    all_genres = sorted(set(genre for sublist in all_data['genre_list'] for genre in sublist))
    genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
    num_genres = len(all_genres)
    all_data['genre_embedding'] = all_data['genre_list'].apply(lambda x: encode_genre_list(x, genre_to_idx, num_genres))
    all_data.drop(columns=['genre_list'], inplace=True)
    return all_data, genre_to_idx

def encode_genre_list(genre_list, genre_to_idx, num_genres):
    multi_hot = np.zeros(num_genres, dtype=int)
    for genre in genre_list:
        if genre in genre_to_idx:
            multi_hot[genre_to_idx[genre]] = 1
    return multi_hot

def label_encode(all_data):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    all_data['user_encoded'] = user_encoder.fit_transform(all_data['user'])
    all_data['item_encoded'] = item_encoder.fit_transform(all_data['item'])
    columns = ['user_encoded', 'item_encoded'] + [col for col in all_data.columns if col not in ['user_encoded', 'item_encoded']]
    all_data = all_data[columns]
    return all_data, user_encoder, item_encoder