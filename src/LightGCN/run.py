import torch
import pandas as pd
import os
from .train import train_model, get_predictions
from .utils import create_submission
from .preprocess import preprocess_data


def main(args):
    """
    LightGCN 모델의 전체 실행 과정을 관리하는 메인 함수입니다.

    이 함수는 다음과 같은 단계를 수행합니다:
    1. 필요한 경우 데이터 전처리
    2. 전처리된 데이터 로드
    3. LightGCN 모델 학습
    4. 사용자별 추천 생성
    5. 추천 결과를 CSV 파일로 저장

    매개변수:
    args: 실행에 필요한 모든 설정을 포함하는 객체
        - model_args: 모델 관련 인자
        - dataset: 데이터셋 관련 경로 정보

    주요 처리 과정:
    - 전처리 여부 확인 및 실행
    - 데이터 로드 및 분할 설정
    - 모델 학습 및 추천 생성
    - 결과 파일 생성

    이 함수는 별도의 반환값이 없으며, 처리 결과를 파일로 저장합니다.
    """
    model_args=args.model_args
    # 전처리 수행
    if not model_args.preprocessed:
        print("Starting preprocessing...")
        input_file_path = os.path.join(args.dataset.data_path, "train_ratings.csv")
        test_size = model_args.test_size
        random_state = model_args.random_state
        preprocess_data(input_file_path, args.dataset.preprocessing_path, test_size, random_state)
        print("Preprocessing completed.")

    # 데이터 로드
    print("Loading data...")
    train_data = pd.read_csv(os.path.join(args.dataset.preprocessing_path, "processed_train_data.csv"))
    val_data = pd.read_csv(os.path.join(args.dataset.preprocessing_path, "processed_val_data.csv"))
    full_data = pd.concat([train_data, val_data])
    user_id_map = pd.read_csv(os.path.join(args.dataset.preprocessing_path, "user_id_map.csv"))
    item_id_map = pd.read_csv(os.path.join(args.dataset.preprocessing_path, "item_id_map.csv"))

    if not model_args.data_split:
        train_data = full_data
        val_data = None

    # 모델 학습
    model, adj_matrix, model_name = train_model(
        train_data=train_data,
        val_data=val_data,
        n_layers=model_args.n_layers,
        embedding_dim=model_args.embedding_dim,
        batch_size=model_args.batch_size,
        n_epochs=model_args.n_epochs,
        patience=model_args.patience,
        lr=model_args.learning_rate
    )

    # 사용자별 시청 기록 생성
    user_interactions = full_data.groupby("user")["item"].apply(set).to_dict()

    # 추천 생성
    k=model_args.top_k
    recommendations = get_predictions(
        model, adj_matrix, len(user_id_map), user_interactions, k
    )

    # 제출 파일 생성
    submission_path = os.path.join(args.dataset.output_path, f"{model_name}_submission.csv")
    create_submission(recommendations, user_id_map, item_id_map, submission_path)
    print(f"Submission file created: {submission_path}")