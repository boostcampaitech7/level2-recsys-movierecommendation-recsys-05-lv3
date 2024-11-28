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

user_recommendations = {}

for df in dataframes:
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

final_df = pd.DataFrame(
    [(user, item) for user, items in final_recommendations.items() for item in items],
    columns=["user", "recommended_item"]
)

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

output_file = os.path.join(output_final_dir, f'{timestamp}.csv')

final_df.to_csv(output_file, index=False)

print(f"추천 결과가 {output_file}에 저장되었습니다.")
