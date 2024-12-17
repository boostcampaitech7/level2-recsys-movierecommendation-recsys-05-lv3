import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def load_data(data_path):
    """
    CSV 파일에서 데이터를 로드하고 'watched' 열을 추가합니다.

    Args:
        data_path: 데이터 파일 경로.

    Returns:
        DataFrame: 로드된 데이터프레임.
    """
    train_df = pd.read_csv(data_path)
    train_df['watched'] = 1
    return train_df

def encode_data(train_df):
    """
    사용자와 아이템을 정수로 인코딩합니다.

    Args:
        train_df: 원본 데이터프레임.

    Returns:
        tuple: 인코딩된 데이터프레임, 사용자 인코더, 아이템 인코더.
    """
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    train_df['user_id'] = user_encoder.fit_transform(train_df['user'])
    train_df['item_id'] = item_encoder.fit_transform(train_df['item'])
    return train_df, user_encoder, item_encoder

def generate_negative_samples(train_df, num_items, max_negative_per_user=10, negative_ratio=0.5, random_seed=42):
    """
    부정 샘플을 생성합니다.

    Args:
        train_df: 인코딩된 데이터프레임.
        num_items: 총 아이템 수.
        max_negative_per_user: 사용자당 최대 부정 샘플 수.
        negative_ratio: 부정 샘플 비율.
        random_seed: 난수 시드.

    Returns:
        DataFrame: 부정 샘플 데이터프레임.
    """
    np.random.seed(random_seed)
    users = train_df['user_id'].unique()
    positive_interactions = set(zip(train_df['user_id'], train_df['item_id']))
    all_items = set(range(num_items))
    negative_samples = []

    for user in tqdm(users, desc="Generating negative samples", unit="user"):
        user_items = train_df[train_df['user_id'] == user]['item_id'].tolist()
        num_user_items = len(user_items)
        non_interacted_items = list(all_items - set(user_items))

        if num_user_items <= 500:
            num_negative = int(num_user_items * negative_ratio)
        else:
            num_negative = max_negative_per_user

        sampled_items = np.random.choice(non_interacted_items, size=num_negative, replace=False)
        for item in sampled_items:
            if (user, item) not in positive_interactions:
                negative_samples.append([user, item, 0])

    negative_df = pd.DataFrame(negative_samples, columns=['user_id', 'item_id', 'watched'])
    return negative_df

def prepare_final_data(train_df, negative_df):
    """
    긍정 및 부정 샘플을 결합하여 최종 데이터를 준비합니다.

    Args:
        train_df: 긍정 샘플 데이터프레임.
        negative_df: 부정 샘플 데이터프레임.

    Returns:
        DataFrame: 최종 데이터프레임.
    """
    final_data = pd.concat([train_df, negative_df])
    final_data.reset_index(drop=True, inplace=True)
    return final_data