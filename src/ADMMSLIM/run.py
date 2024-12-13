import os
import tqdm
import pandas as pd
from .preprocessing import load_and_preprocess_data
from .model import AdmmSlim
from .inference import generate_recommendations

def main(args):
    # 데이터 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, args.dataset.data_path)
    # 데이터 로드 및 전처리
    train_df, user_item_matrix, user_encoder, item_encoder = load_and_preprocess_data(data_path)

    # 모델 생성 및 학습
    model = AdmmSlim(lambda_1=args.model_args.lambda_1, 
                     lambda_2=args.model_args.lambda_2, 
                     rho=args.model_args.rho, 
                     positive=args.model_args.positive, 
                     n_iter=args.model_args.n_iter, 
                     verbose=args.model_args.verbose)
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