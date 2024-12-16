import torch
import pandas as pd
import os
from .train import train_model, get_predictions
from .utils import create_submission
from .preprocess import preprocess_data


def main(args):
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