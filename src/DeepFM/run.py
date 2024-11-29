import torch
from torch.utils.data import DataLoader
from preprocessing import *
from model import DeepFM
from train import *
from inference import recommend_top_k

# 데이터 경로 설정
data_path = './data/train'

# 데이터 로드 및 전처리
train_df, year_data, title_data, genre_data = load_data(data_path)
all_data = preprocess_data(train_df, title_data, year_data)
fasttext_model = train_fasttext(all_data)
all_data['title_embedding'] = all_data['title_tokens'].apply(lambda x: get_title_embedding(x, fasttext_model, 50))
all_data, genre_to_idx = encode_genres(all_data, genre_data)
all_data, user_encoder, item_encoder = label_encode(all_data)
all_items = all_data['item_encoded'].unique()
item_counts = all_data['item_encoded'].value_counts()
item_probs = item_counts / item_counts.sum()

# 모델 및 데이터셋 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
user_dim = all_data['user_encoded'].nunique()
item_dim = all_data['item_encoded'].nunique()
input_dims = [user_dim, item_dim]
embedding_dim = 32
mlp_dims = [64, 32, 16]
drop_rate = 0.1
user_item_embeddings = np.random.rand(len(all_data['user_encoded'].unique()) + len(all_items), embedding_dim)
model = DeepFM(input_dims=input_dims, embedding_dim=embedding_dim, mlp_dims=mlp_dims, drop_rate=drop_rate).to(device)

# 데이터셋 및 DataLoader 생성
user_item_dict = all_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()
user_col = torch.tensor(all_data['user_encoded'].values, dtype=torch.long).to(device)
item_col = torch.tensor(all_data['item_encoded'].values, dtype=torch.long).to(device)
genre_col = torch.tensor(all_data['genre_embedding'].tolist(), dtype=torch.float32).to(device)
all_years = all_data['year'].values
dataset = PairwiseRecommendationDataset(
    user_col=user_col,
    item_col=item_col,
    user_item_dict=user_item_dict,
    all_items=all_items,
    item_probs=item_probs,
    user_item_embeddings=user_item_embeddings,
    num_negatives=10
)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 모델 학습
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
train_pairwise(model, train_loader, optimizer, epochs=10)

# 추천 수행
recommendations = recommend_top_k(model, user_item_dict, all_items, k=10)

# 추천 결과 저장
user_id = dict(zip(all_data['user_encoded'], all_data['user']))
item_id = dict(zip(all_data['item_encoded'], all_data['item']))
recommendation_list = []
for user_encoded, item_encoded_list in recommendations.items():
    user = user_id[user_encoded]
    for item_encoded in item_encoded_list:
        item = item_id[item_encoded]
        recommendation_list.append({'user': user, 'item': item})
recommendations_df = pd.DataFrame(recommendation_list)
recommendations_df.to_csv('recommendations.csv', index=False)
print("추천 결과가 'recommendations.csv' 파일로 저장되었습니다.")