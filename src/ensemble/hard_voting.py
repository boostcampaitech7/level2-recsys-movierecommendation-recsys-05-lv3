import pandas as pd
import os
from collections import Counter
from datetime import datetime

output_dir = './predictions'
output_final_dir = './finals'

if not os.path.exists(output_final_dir):
    os.makedirs(output_final_dir)

file_paths = sorted([os.path.join(output_dir, file) for file in os.listdir(output_dir) if file.endswith('.csv')])
dataframes = [pd.read_csv(file) for file in file_paths]

# 사용자에게 동일 가중치 사용 여부 확인
while True:
    user_input = input("모든 파일에 동일한 가중치를 적용하시겠습니까? (y/n): ").lower()
    if user_input in ['y', 'n']:
        use_equal_weights = (user_input == 'y')
        break
    else:
        print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")

# 가중치 입력 부분
weights = {}
if use_equal_weights:
    equal_weight = 1.0 / len(file_paths)
    weights = {os.path.basename(file): equal_weight for file in file_paths}
else:
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"파일: {file_name}")
        weight = float(input(f"{file_name}의 가중치를 입력하세요: "))
        weights[file_name] = weight

# 데이터프레임 로드 및 가중치 적용
dataframes = [(pd.read_csv(file), weights[os.path.basename(file)]) for file in file_paths]

if use_equal_weights:
    # 동일 가중치 적용 시 기존 코드 사용
    user_recommendations = {}
    for df, _ in dataframes:
        for _, row in df.iterrows():
            user = row[0]
            item = row[1]
            if user not in user_recommendations:
                user_recommendations[user] = []
            user_recommendations[user].append(item)

    final_recommendations = {}
    for user, items in user_recommendations.items():
        item_counts = Counter(items)
        most_common_items = [item for item, _ in item_counts.most_common(10)]
        final_recommendations[user] = most_common_items

else:
    # 다른 가중치 적용 시 수정된 코드 사용
    user_recommendations = {}
    for df, weight in dataframes:
        for _, row in df.iterrows():
            user = row[0]
            item = row[1]
            if user not in user_recommendations:
                user_recommendations[user] = Counter()
            user_recommendations[user][item] += weight

    final_recommendations = {}
    for user, items in user_recommendations.items():
        final_recommendations[user] = [item for item, _ in items.most_common(10)]

final_df = pd.DataFrame(
    [(user, item) for user, items in final_recommendations.items() for item in items],
    columns=["user", "recommended_item"]
)

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

output_file = os.path.join(output_final_dir, f'{timestamp}.csv')

final_df.to_csv(output_file, index=False)

print(f"추천 결과가 {output_file}에 저장되었습니다.")
