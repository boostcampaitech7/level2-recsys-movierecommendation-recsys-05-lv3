import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from numpy import cumsum
from torch.utils.data import DataLoader, Dataset
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score



# random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Rating Data Load
print("--Rating Data Load--")

raw_rating_df = pd.read_csv("data/train/train_ratings.csv")

raw_rating_df['rating'] = 1.0 # implicit feedback
raw_rating_df.drop(['time'],axis=1,inplace=True)

users = raw_rating_df['user'].unique()
items = raw_rating_df['item'].unique()


# Director Data Load
print("--Director Data Load--")
raw_director_df = pd.read_csv("data/train/directors.tsv", sep='\t')
## label encoding
director_encoder = LabelEncoder()
raw_director_df['encoded_director'] = director_encoder.fit_transform(raw_director_df['director'])
## layer 생성
embedding_dim = 10
director_embedding = nn.Embedding(raw_director_df['encoded_director'].nunique(), embedding_dim)
## item per director
director_grouped = raw_director_df.groupby('item')['encoded_director'].apply(list).sort_index()
director_avg_vectors = []
## embedding
for ids in director_grouped:
    director_embeds = director_embedding(torch.tensor(ids))
    avg_vector = torch.mean(director_embeds, dim=0)
    director_avg_vectors.append(avg_vector)

director_avg_vectors = [vec.detach().numpy() for vec in director_avg_vectors]

director_ = np.concatenate([director_grouped.index.to_numpy().reshape(-1, 1), director_avg_vectors], axis=1)
        
director_df = pd.DataFrame({'item': director_[:, 0]})
director_df['director_vectors'] = list(director_[:, 1:].astype(np.float32))


# Writer Data Load
print("--Writer Data Load--")
raw_writer_df = pd.read_csv("data/train/writers.tsv", sep='\t')
## label encoding
writer_encoder = LabelEncoder()
raw_writer_df['encoded_writer'] = writer_encoder.fit_transform(raw_writer_df['writer'])
## layer 생성
embedding_dim = 10
writer_embedding = nn.Embedding(raw_writer_df['encoded_writer'].nunique(), embedding_dim)
## item per director
writer_grouped = raw_writer_df.groupby('item')['encoded_writer'].apply(list).sort_index()
writer_avg_vectors = []
## embedding
for ids in writer_grouped:
    writer_embeds = writer_embedding(torch.tensor(ids))
    avg_vector = torch.mean(writer_embeds, dim=0)
    writer_avg_vectors.append(avg_vector)

writer_avg_vectors = [vec.detach().numpy() for vec in writer_avg_vectors]

writer_ = np.concatenate([writer_grouped.index.to_numpy().reshape(-1, 1), writer_avg_vectors], axis=1)
        
writer_df = pd.DataFrame({'item': writer_[:, 0]})
writer_df['writer_vectors'] = list(writer_[:, 1:].astype(np.float32))


# Year Data Load
print("--Year Data Load--")
raw_year_df = pd.read_csv("data/train/years.tsv", sep='\t')
year_df = raw_year_df


# Genre Data Load
print("--Genre Data Load--")
raw_genre_df = pd.read_csv("data/train/genres.tsv", sep='\t')

genre_multi_hot = raw_genre_df.groupby('item')['genre'].apply(
    lambda x: np.array([1 if genre in x.tolist() else 0 for genre in raw_genre_df['genre'].unique()]))

genre_ = np.concatenate([genre_multi_hot.index.to_numpy().reshape(-1, 1),
                            np.array(genre_multi_hot).reshape(-1,1)], axis=1)

genre_df = pd.DataFrame(genre_, columns=['item','genre_vectors'])


# Negative instance 생성
print("--Create Negative instances--")
rating_df = pd.read_csv("rating_df_50.csv")
rating_df.drop(columns=['Unnamed: 0'], inplace=True)

# negative_samples = []
# for user in tqdm(users):
#     # 시청한 영화
#     positive_items = set(raw_rating_df[raw_rating_df['user'] == user]['item'])
#     # 시청하지 않은 영화
#     negative_items = set(raw_rating_df['item'].unique()) - positive_items
#     # Negative Sample 개수는 시청한 영화 개수와 동일
#     # num_negatives = len(positive_items)
#     num_negatives = 50
#     sampled_negatives = np.random.choice(list(negative_items), num_negatives, replace=False)
#     for neg_movie in sampled_negatives:
#         negative_samples.append({'user': user, 'item': neg_movie, 'rating': 0})

# negative_df = pd.DataFrame(negative_samples)
# rating_df = pd.concat([raw_rating_df, negative_df], ignore_index=True)


# Merge Datas
print("--Merge Datas--")
merged_df = pd.merge(rating_df, director_df, how='left', on='item')
merged_df = pd.merge(merged_df, writer_df, how='left', on='item')
merged_df = pd.merge(merged_df, year_df, how='left', on='item')
merged_df = pd.merge(merged_df, genre_df, how='left', on='item')


# 리스트 형태의 임베딩 데이터를 넘파이 배열로 변환
merged_df['director_vectors'] = merged_df['director_vectors'].apply(lambda x: np.array(x, dtype=np.float32))
merged_df['writer_vectors'] = merged_df['writer_vectors'].apply(lambda x: np.array(x, dtype=np.float32))
merged_df['genre_vectors'] = merged_df['genre_vectors'].apply(lambda x: np.array(x, dtype=np.float32))


# 결측치 제거
print("--Fill na--")
merged_df['director_vectors'] = merged_df['director_vectors'].apply(
    lambda x: np.zeros(10) if not isinstance(x, np.ndarray) else x)
merged_df['writer_vectors'] = merged_df['writer_vectors'].apply(
    lambda x: np.zeros(10) if not isinstance(x, np.ndarray) else x)
merged_df['genre_vectors'] = merged_df['genre_vectors'].apply(
    lambda x: np.zeros(18) if not isinstance(x, np.ndarray) else x)
merged_df['year'] = merged_df['year'].fillna(0).astype(np.float32)


# 필요한 열을 합쳐서 하나의 피처 배열 생성
print("--Feature Vector--")
feature_vectors = []
for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing feature vectors"):
    feature_vector = np.concatenate([
        np.array([row['user']]),               # user ID
        np.array([row['item']]),               # item ID
        row['director_vectors'],               # 감독 벡터
        row['writer_vectors'],                 # 작가 벡터
        np.array([row['year']]),               # 연도 정보
        row['genre_vectors']                   # 장르 벡터
    ])
    feature_vectors.append(feature_vector)
merged_df['feature_vectors'] = feature_vectors

# 
class Text_Dataset(Dataset):
    def __init__(self, feature_vectors, rating=None):
        self.feature_vectors = np.stack(feature_vectors)
        self.rating = rating

    def __len__(self):
        return len(self.feature_vectors)
    
    def __getitem__(self, i):
        data = {
            'features': torch.tensor(self.feature_vectors[i], dtype=torch.float32),
        }
        if self.rating is not None:
            data['rating'] = torch.tensor(self.rating[i], dtype=torch.float32)
        return data
        '''
        return {
                'features' : torch.tensor(self.feature_vectors[i], dtype=torch.float32),
                'rating' : torch.tensor(self.rating[i], dtype=torch.float32),
                } if self.rating is not None else \
                {'features' : torch.tensor(self.feature_vectors[i], dtype=torch.float32)}
        '''


# Data Processing
def data_processing(data):
    sparse_cols = ['user','item']

    # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        data[col] = data[col].fillna('unknown')
        unique_labels = data[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        data[col] = data[col].map(label2idx[col])

    field_dims = [len(label2idx[col]) for col in sparse_cols]

    data = {
            'train':data,
            'field_names':sparse_cols,
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            }
    
    return data


# Data Split
def data_split(data, valid_ratio=0.1):
    
    if valid_ratio == 0:
        data['X_train'] = data['train'].drop('rating', axis=1)
        data['y_train'] = data['train']['rating']

    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
                                                            data['train'].drop(['rating'], axis=1),
                                                            data['train']['rating'],
                                                            test_size=valid_ratio,
                                                            shuffle=True
                                                            )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    return data


# Data loader
def data_loader(data, batch_size, shuffle, valid_ratio=0.1):
   
    train_features = data['X_train']['feature_vectors']
    train_ratings = data['y_train'].values
    train_dataset = Text_Dataset(train_features, train_ratings)

    valid_dataset = None
    if valid_ratio != 0:
        valid_features = data['X_valid']['feature_vectors']
        valid_ratings = data['y_valid'].values
        valid_dataset = Text_Dataset(valid_features, valid_ratings)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) if valid_ratio else None

    data['train_dataloader'], data['valid_dataloader'] = train_dataloader, valid_dataloader
    
    return data


# FM
class FeaturesLinear(nn.Module):
    def __init__(self, field_dims:list, output_dim:int=1, bias:bool=True):
        super().__init__()
        self.feature_dims = sum(field_dims)
        self.output_dim = output_dim
        self.offsets = [0, *cumsum(field_dims)[:-1]]

        self.fc = nn.Embedding(self.feature_dims, self.output_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty((self.output_dim,)), requires_grad=True)
    
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start
            if isinstance(m, nn.Parameter):
                nn.init.constant_(m, 0)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(self.fc(x), dim=1) + self.bias if hasattr(self, 'bias') \
                else torch.sum(self.fc(x), dim=1)
    

class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims:list, embed_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = [0, *cumsum(field_dims)[:-1]]
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return self.embedding(x)  # (batch_size, num_fields, embed_dim)
    

class FMLayer_Dense(nn.Module):
    def __init__(self):
        super().__init__()

    def square(self, x:torch.Tensor):
        return torch.pow(x,2)

    def forward(self, x):
        # square_of_sum =   # FILL HERE : Use `torch.sum()` and `self.square()` #
        # sum_of_square =   # FILL HERE : Use `torch.sum()` and `self.square()` #
        square_of_sum = self.square(torch.sum(x, dim=1))
        sum_of_square = torch.sum(self.square(x), dim=1)
        
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)


class FMLayer_Sparse(nn.Module):
    def __init__(self, field_dims:list, factor_dim:int):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, factor_dim)
        self.fm = FMLayer_Dense()


    def square(self, x):
        return torch.pow(x,2)
    

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.fm(x)
        
        return x
    

class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.embedding = nn.Embedding(input_dim+1, embed_dim)
        self.fm = FMLayer_Dense()
    def forward(self, x):
        x = torch.clamp(x, min=0, max=self.embedding.num_embeddings - 1)
        linear_part = self.linear(x)  # 선형 항 계산
        embedded_x = self.embedding(x.long())  # 임베딩 계산
        fm_part = self.fm(embedded_x)  # FM 상호작용 항
        return linear_part.squeeze(1) + fm_part
    

# Model Train
def train(model, epochs, dataloader, criterion, optimizer, device, valid_ratio):

    for epoch in range(epochs):
        model.train() # 모델 학습 모드로 변경
        total_loss, train_len = 0, len(dataloader['train_dataloader'])

        for data in tqdm(dataloader['train_dataloader'], desc=f'[Epoch {epoch+1:02d}/{epochs:02d}]'):
            x, y = data["features"].to(device).float(), data["rating"].to(device).float()
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        msg = ''
        train_loss = total_loss / train_len
        msg += f'\tTrain Loss : {train_loss:.3f}'

        if valid_ratio != 0:  # valid 데이터가 존재할 경우
            valid_loss = valid(model, dataloader['valid_dataloader'], criterion, device)
            msg += f'\n\tValid Loss : {valid_loss:.3f}'
            print(msg)
        else:  # valid 데이터가 없을 경우
            print(msg)
        
    return model


def valid(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    for data in dataloader:
        x, y = data["features"].to(device).float(), data["rating"].to(device).float()
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def test(model, data_loader, criterion, device):
    num_batches = len(data_loader)
    test_loss, y_all, pred_all = 0, list(), list()

    with torch.no_grad():
        for data in data_loader:
            x, y = data[0].to(device).float(), data[1].to(device).float()
            pred = model(x)
            test_loss += criterion(pred, y).item() / num_batches
            y_all.append(y)
            pred_all.append(pred)

    y_all = torch.cat(y_all).cpu()
    pred_all = torch.cat(pred_all).cpu()

    err = roc_auc_score(y_all, torch.sigmoid(pred_all)).item()
    print(f"Test Error: \n  AUC: {err:>8f} \n  Avg loss: {test_loss:>8f}")

    return err, test_loss




######## Hyperparameter ########
batch_size = 1024 # 배치 사이즈
shuffle = True
embed_dim = 4 # embed feature의 dimension
epochs = 1 # epoch 돌릴 횟수
learning_rate = 0.0001 # 학습이 반영되는 정도를 나타내는 파라미터
weight_decay=1e-4 # 정규화를 위한 파라미터
# input_dim = raw_rating_df.shape[1] - 1
valid_ratio = 0.1




print("--Data Processing--")
data_processing_df = data_processing(merged_df)

print("--Data Split--")
data_split_df = data_split(data_processing_df)

print("--Data Loader--")
data_loader_df = data_loader(data_split_df, batch_size, shuffle, 0.1)

print("--Model Load--")
input_dim = len(merged_df['feature_vectors'][0])
model = FactorizationMachine(input_dim, embed_dim).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, amsgrad=False, weight_decay=weight_decay)

print("--Training--")
model = train(model, epochs, data_loader_df, criterion, optimizer, device, valid_ratio)

print("Done")