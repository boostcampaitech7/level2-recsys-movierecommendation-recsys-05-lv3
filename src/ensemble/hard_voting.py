import yaml
import pandas as pd
import os
from collections import Counter

def load_config(yaml_file):
    """
    YAML 설정 파일을 로드합니다.

    Args:
        yaml_file (str): YAML 설정 파일의 경로

    Returns:
        dict: YAML 파일의 내용을 담은 딕셔너리
    """
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def weighted_vote(predictions, weights):
    """
    가중치를 적용한 투표를 수행하여 상위 10개 아이템을 선택합니다.

    Args:
        predictions (dict): 모델별 예측 결과
        weights (dict): 모델별 가중치 정보

    Returns:
        list: 가중치 투표 결과 상위 10개 아이템 목록
    """
    weighted_votes = Counter()
    for model, preds in predictions.items():
        if weights[model]['use']:
            for item in preds:
                weighted_votes[item] += weights[model]['weight']
    return [item for item, _ in weighted_votes.most_common(10)]

def main():
    """
    메인 함수: 설정을 로드하고, 예측 결과를 처리하여 최종 추천을 생성합니다.
    
    - YAML 설정 파일을 로드합니다.
    - 각 모델의 예측 결과를 읽어 처리합니다.
    - 가중치 투표를 통해 최종 추천을 생성합니다.
    - 최종 추천 결과를 CSV 파일로 저장합니다.
    - 사용된 모델과 가중치 정보를 로그 파일로 저장합니다.

    오류 발생 시 프로그램을 종료합니다.
    """
    config = load_config('../../config/model_weights.yaml')
    output_dir = config['output_dir']
    model_weights = config['model_weights']
    experiment_id = config['id']
    
    predictions = {}
    used_models = {}
    for model, info in model_weights.items():
        if info['use']:
            file_path = os.path.join(output_dir, info['file'])
            try:
                df = pd.read_csv(file_path)
                predictions[model] = df.groupby('user')['item'].apply(list).to_dict()
                used_models[model] = info['weight']
            except FileNotFoundError:
                print(f"Error: File not found for model {model}: {file_path}")
                return

    if not predictions:
        print("Error: No valid prediction files found. Exiting.")
        return

    final_recommendations = {}
    for user in predictions[list(predictions.keys())[0]]:
        user_preds = {model: preds[user] for model, preds in predictions.items() if user in preds}
        final_recommendations[user] = weighted_vote(user_preds, model_weights)

    final_df = pd.DataFrame(
        [(user, item) for user, items in final_recommendations.items() for item in items],
        columns=["user", "item"]
    )
    
    final_output_path = os.path.join(output_dir, f'final_recommendations_{experiment_id}.csv')
    final_df.to_csv(final_output_path, index=False)
    print(f"Final recommendations saved to {final_output_path}")

    # 로그 파일 생성
    log_output_path = os.path.join(output_dir, f'final_recommendations_{experiment_id}.log')
    with open(log_output_path, 'w') as log_file:
        log_file.write(f"Experiment ID: {experiment_id}\n\n")
        log_file.write("Used models and their weights:\n")
        for model, weight in used_models.items():
            log_file.write(f"{model}: {weight}\n")

    print(f"Log file created: {log_output_path}")

if __name__ == "__main__":
    main()
