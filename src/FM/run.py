import torch
import numpy as np
import random

from model import *
from preprocessing import *
from train import *
from inference import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="../../../../data/train/")

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--shuffle', type=bool, default=True)


    return parser.parse_args()


args = parse_args()

batch_size = args.batch_size
epochs = args.epochs
embed_dim = args.embed_dim
learning_rate = args.learning_rate
weight_decay = args.weight_decay
valid_ratio = args.valid_ratio
shuffle = args.shuffle
seed = args.seed

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



print("--Data Prepare--")
raw_rating_df = load_rating_data(args.path+"train_ratings.csv")
rating_df = negative_sampling(raw_rating_df)

raw_director_df = load_director_data(args.path+"directors.tsv")
raw_writer_df = load_writer_data(args.path+"writers.tsv")
raw_year_df = load_year_data(args.path+"years.tsv")
raw_genre_df = load_genre_data(args.path+"genres.tsv")

users = raw_rating_df['user'].unique()
items = raw_rating_df['item'].unique()


print("--Merge Datas--")
merged_df = pd.merge(rating_df, raw_director_df, how='left', on='item')
merged_df = pd.merge(merged_df, raw_writer_df, how='left', on='item')
merged_df = pd.merge(merged_df, raw_year_df, how='left', on='item')
merged_df = pd.merge(merged_df, raw_genre_df, how='left', on='item')

# 리스트 형태의 임베딩 데이터를 넘파이 배열로 변환
merged_df['director_vectors'] = merged_df['director_vectors'].apply(lambda x: np.array(x, dtype=np.float32))
merged_df['writer_vectors'] = merged_df['writer_vectors'].apply(lambda x: np.array(x, dtype=np.float32))
merged_df['genre_vectors'] = merged_df['genre_vectors'].apply(lambda x: np.array(x, dtype=np.float32))

merged_df = fill_na_vectors(merged_df)
train_df = create_feature_vectors_for_train(merged_df)
inference_df = create_feature_vectors_for_inference(merged_df)



######## Hyperparameter ########
batch_size = 128     
shuffle = True          
embed_dim = 4         
epochs = 1            
learning_rate = 0.0001 
weight_decay=1e-4       
valid_ratio = 0.1



print("--Data Processing--")
data_processing_df = data_processing(train_df)

print("--Data Split--")
data_split_df = data_split(data_processing_df)

print("--Data Loader--")
data_loader_df = data_loader(data_split_df, batch_size, shuffle, 0.1)

print("--Model Load--")
input_dim = len(train_df['feature_vectors'][0])
model = FactorizationMachine(input_dim).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, amsgrad=False, weight_decay=weight_decay)

print("--Training--")
model = train(model, epochs, data_loader_df, criterion, optimizer, device, valid_ratio)

print("--Inference--")
recommendation_df = generate_recommendation(model, inference_df, device, 10)
recommendation_df = recommendation_df.sort_values('user')

print("--Submission Data--")
recommendation_df.to_csv("../../saved/FM.csv", index=False)