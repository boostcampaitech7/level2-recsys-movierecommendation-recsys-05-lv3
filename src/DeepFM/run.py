import os
import pandas as pd
import numpy as np
import re
from gensim.models import FastText
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from dataset import *
from inference import * 
from model import * 
from preprocessing import * 
from train import *



def main(data_path='./data/train'):
    print("데이터 준비------------------")
    
    train_df, year_df, title_df, genre_df = load_data(data_path)
    train_df_processed = preprocess_train_data(train_df)
    
    merged_all_df = merge_data(train_df_processed, title_df, year_df)
    
    cleaned_all_df = clean_titles(merged_all_df)
    tokenized_all_df = tokenize_titles(cleaned_all_df)
    fasttext_model_trained = train_fasttext_model(tokenized_all_df)
    embedded_all_df = add_title_embeddings(tokenized_all_df, fasttext_model_trained, 50)
    genres_per_item_processed = process_genres(genre_df)
    merged_with_genres_all_df = merge_genres(embedded_all_df, genres_per_item_processed)
    _, final_all_df_with_embeddings = add_genre_embeddings(merged_with_genres_all_df)
    user_id_map_final, item_id_map_final, final_prepared_all_df = label_encode_users_items(final_all_df_with_embeddings)

    all_data = final_prepared_all_df

    print("모델 설정------------------")

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # 각 범주형 변수의 고유값 개수
    user_dim = all_data['user_encoded'].nunique()
    item_dim = all_data['item_encoded'].nunique()


    # 범주형 변수의 고유값 개수를 리스트로 전달
    input_dims = [user_dim, item_dim] 

    # 임베딩 차원 설정
    embedding_dim = 32

    # MLP 레이어 차원 설정
    mlp_dims = [64, 32, 16]

    # 드롭아웃 비율 설정
    drop_rate = 0.1

    # DeepFM 모델 초기화
    model = DeepFM(input_dims=input_dims, 
                embedding_dim=embedding_dim, mlp_dims=mlp_dims, drop_rate=drop_rate).to(device)



    print("모델 학습 준비------------------")

    # 유저별 시청 아이템 딕셔너리 생성
    user_item_dict = all_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()

    # Categorical Features (long 타입)
    user_col = torch.tensor(all_data['user_encoded'].values, dtype=torch.long).to(device)
    item_col = torch.tensor(all_data['item_encoded'].values, dtype=torch.long).to(device)

    # Continuous Features (float 타입)
    genre_col = torch.tensor(all_data['genre_embedding'].tolist(), dtype=torch.float32).to(device)

    # 'year'를 numpy 배열로 미리 준비
    all_years = all_data['year'].values

    # DataLoader 생성
    dataset = PairwiseRecommendationDataset(
        user_col=user_col,
        item_col=item_col,
        user_item_dict=user_item_dict,
        all_items=all_data,
        item_probs=item_probs,
        user_item_embeddings=user_item_embeddings,
        num_negatives=10  # Negative 샘플 수
    )
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


    print("모델 학습------------------")

    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 모델 학습
    train_pairwise(model, train_loader, optimizer, device, genre_col, all_years, epochs=10)


    print("k개 추천------------------")

    #k개 추천
    # 1. 인코딩된 ID를 원래의 ID로 매핑하기 위한 딕셔너리 생성
    user_id = dict(zip(all_data['user_encoded'], all_data['user']))
    item_id = dict(zip(all_data['item_encoded'], all_data['item']))

    # 2. 추천 결과를 원래의 ID로 복원하여 데이터프레임 생성
    recommendation_list = []

    recommendations = recommend_top_k(model, user_item_dict, all_items, k=10)

    for user_encoded, item_encoded_list in recommendations.items():
        # 원래의 유저 ID로 변환
        user = user_id[user_encoded]
        
        for item_encoded in item_encoded_list:
            # 원래의 아이템 ID로 변환
            item = item_id[item_encoded]
            
            # 추천 결과 추가
            recommendation_list.append({
                'user': user,
                'item': item
            })

    # 추천 결과를 데이터프레임으로 변환
    recommendations_df = pd.DataFrame(recommendation_list)

    print("저장------------------")

    # 3. CSV 파일로 저장
    recommendations_df.to_csv('recommendations.csv', index=False)
    print("추천 결과가 'recommendations.csv' 파일로 저장되었습니다.")