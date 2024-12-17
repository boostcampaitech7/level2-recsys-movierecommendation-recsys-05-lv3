import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data(data_path, preprocessing_path, test_size=0.2, random_state=42):
    """
    주어진 데이터를 전처리하고 훈련 및 검증 데이터셋으로 분할합니다.

    매개변수:
    data_path (str): 원본 데이터 파일의 경로
    preprocessing_path (str): 전처리된 데이터를 저장할 경로
    test_size (float): 검증 데이터셋의 비율 (기본값: 0.2)
    random_state (int): 랜덤 시드 값 (기본값: 42)

    처리 과정:
    1. CSV 파일에서 데이터를 로드합니다.
    2. 'time' 열을 제거합니다.
    3. 사용자와 아이템 ID를 연속된 정수로 매핑합니다.
    4. 데이터를 훈련셋과 검증셋으로 분할합니다.
    5. 전처리된 데이터와 ID 매핑 정보를 저장합니다.
    6. 데이터 통계를 출력합니다.

    반환값:
    tuple: (train_data, val_data, user_id_map, item_id_map)
        - train_data (pd.DataFrame): 훈련 데이터셋
        - val_data (pd.DataFrame): 검증 데이터셋
        - user_id_map (dict): 원본 사용자 ID와 매핑된 ID의 딕셔너리
        - item_id_map (dict): 원본 아이템 ID와 매핑된 ID의 딕셔너리
    """
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