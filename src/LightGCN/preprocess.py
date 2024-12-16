import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(data_path, preprocessing_path, test_size=0.2, random_state=42):
    # 데이터 로드
    df = pd.read_csv(data_path)
    
    # 시간 컬럼 제거 (필요없는 경우)
    df = df.drop('time', axis=1)
    
    # 유저와 아이템 ID를 0부터 시작하는 연속된 정수로 매핑
    user_id_map = {id: i for i, id in enumerate(df['user'].unique())}
    item_id_map = {id: i for i, id in enumerate(df['item'].unique())}
    
    # ID 매핑 적용
    df['user'] = df['user'].map(user_id_map)
    df['item'] = df['item'].map(item_id_map)
    
    # 데이터 분할 (train용)
    train_data, val_data = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['user'])
    
    # 출력 디렉토리 생성
    os.makedirs(preprocessing_path, exist_ok=True)
    
    # 데이터 저장
    train_data.to_csv(os.path.join(preprocessing_path, 'processed_train_data.csv'), index=False)
    val_data.to_csv(os.path.join(preprocessing_path, 'processed_val_data.csv'), index=False)
    
    # ID 매핑 정보 저장
    pd.DataFrame(list(user_id_map.items()), 
                columns=['original_id', 'mapped_id']).to_csv(
                    os.path.join(preprocessing_path, 'user_id_map.csv'), index=False)
    pd.DataFrame(list(item_id_map.items()), 
                columns=['original_id', 'mapped_id']).to_csv(
                    os.path.join(preprocessing_path, 'item_id_map.csv'), index=False)
    
    print(f"Number of users: {len(user_id_map)}")
    print(f"Number of items: {len(item_id_map)}")
    print(f"Number of interactions: {len(df)}")
    print(f"Train set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    return train_data, val_data, user_id_map, item_id_map