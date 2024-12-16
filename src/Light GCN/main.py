import torch
import pandas as pd
import os
from train import train_model, get_predictions
from utils import create_submission
from preprocess import preprocess_data


def main(args):
    
    # 전처리 수행
    if args.preprocessed:
        print("Starting preprocessing...")
        input_file_path = os.path.join(args.dataset.data_path, "train_ratings.csv")
        test_size = args.split.test_size
        random_state = args.split.random_state
        preprocess_data(input_file_path, args.dataset.preprocessing_path, test_size, random_state)
        print("Preprocessing completed.")

    # 데이터 로드
    print("Loading data...")
    train_data = pd.read_csv(os.path.join(args.dataset.output_path, "processed_train_data.csv"))
    val_data = pd.read_csv(os.path.join(args.dataset.output_path, "processed_val_data.csv"))
    full_data = pd.concat([train_data, val_data])
    user_id_map = pd.read_csv(os.path.join(args.dataset.output_path, "user_id_map.csv"))
    item_id_map = pd.read_csv(os.path.join(args.dataset.output_path, "item_id_map.csv"))

    if not args.data_split:
        train_data = full_data
        val_data = None

    # 모델 학습
    model, adj_matrix, model_name = train_model(
        train_data=train_data,
        val_data=val_data,
        n_layers=args.params.n_layers,
        embedding_dim=args.params.embedding_dim,
        batch_size=args.params.batch_size,
        n_epochs=args.params.n_epochs,
        patience=args.params.patience,
        lr=args.learning_rate,
    )

    # 사용자별 시청 기록 생성
    user_interactions = full_data.groupby("user")["item"].apply(set).to_dict()

    # 추천 생성
    k=args.params.top_k
    recommendations = get_predictions(
        model, adj_matrix, len(user_id_map), user_interactions, k
    )

    # 제출 파일 생성
    submission_path = os.path.join(args.dataset.output_path, f"{model_name}_submission.csv")
    create_submission(recommendations, user_id_map, item_id_map, submission_path)
    print(f"Submission file created: {submission_path}")

if __name__ == "__main__":
    args = 
    main(args)