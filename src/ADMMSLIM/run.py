import tqdm
import pandas as pd
from preprocessing import load_and_preprocess_data
from model import AdmmSlim
from inference import generate_recommendations

# 데이터 경로 설정
data_path = 'data/train'

# 데이터 로드 및 전처리
train_df, user_item_matrix, user_encoder, item_encoder = load_and_preprocess_data(data_path)

# 모델 생성 및 학습
model = AdmmSlim(lambda_1=10, lambda_2=100, rho=1000, positive=False, n_iter=10, verbose=True)
model.fit(user_item_matrix)

# 추천 생성
user_recommendations = generate_recommendations(model, user_item_matrix, train_df['user_id'].nunique())

# 추천 결과 저장
recommendations = []

for user_id, item_ids in tqdm(user_recommendations.items(), desc="Saving Recommendations"):
    user_original_id = user_encoder.inverse_transform([user_id])[0]
    for item_id in item_ids:
        item_original_id = item_encoder.inverse_transform([item_id])[0]
        recommendations.append({
            'user': user_original_id,
            'item': item_original_id
        })

recommendations_df = pd.DataFrame(recommendations)
recommendations_df.to_csv('recommendations.csv', index=False)