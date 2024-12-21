import os
import tqdm
import pandas as pd
from .preprocessing import load_and_preprocess_data
from .model import AdmmSlim
from .inference import generate_recommendations

def main(args):
    """
    ADMMSLIM 모델을 사용하여 추천 시스템을 실행하는 메인 함수.

    Args:
        args (Namespace): 설정값을 포함하는 인자. 다음과 같은 정보를 포함해야 합니다:
            - dataset.data_path (str): 데이터셋 경로.
            - model_args.lambda_1 (float): L1 정규화 가중치.
            - model_args.lambda_2 (float): L2 정규화 가중치.
            - model_args.rho (float): 페널티 매개변수.
            - model_args.positive (bool): 계수를 양수로 제한할지 여부.
            - model_args.n_iter (int): 최대 반복 횟수.
            - model_args.verbose (bool): 학습 로그 출력 여부.

    Workflow:
        1. 데이터 로드 및 전처리.
        2. ADMMSLIM 모델 학습.
        3. 사용자별 추천 생성.
        4. 추천 결과를 CSV 파일로 저장.

    Output:
        'recommendations.csv' 파일에 추천 결과를 저장합니다.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, args.dataset.data_path)
    data_path = args.dataset.data_path
    print(args.dataset.data_path)
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