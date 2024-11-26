import numpy as np
import torch
from torch import optim
from copy import deepcopy
from tqdm import tqdm

from dataset import *
from model import *

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="./output")
parser.add_argument('--hidden-dim', type=int, default=600)
parser.add_argument('--latent-dim', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--beta', type=float, default=None)
parser.add_argument('--gamma', type=float, default=0.003)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--n-epochs', type=int, default=100)
parser.add_argument('--n-enc_epochs', type=int, default=3)
parser.add_argument('--n-dec_epochs', type=int, default=1)
parser.add_argument('--not-alternating', default=False, action="store_true")


args = parser.parse_args()

seed = 1337
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data, unique_sid, unique_uid = get_data(args.dataset)  
train_data = data

def generate(batch_size, device, data_in, shuffle=False):
    total_samples = data_in.shape[0]
    idxlist = np.arange(total_samples)

    if shuffle:
        np.random.shuffle(idxlist)

    for st_idx in range(0, total_samples, batch_size):
        end_idx = min(st_idx + batch_size, total_samples)
        idx = idxlist[st_idx:end_idx]
        yield Batch(device, idx, data_in)

class Batch:
    def __init__(self, device, idx, data_in):
        self._device = device
        self._idx = idx
        self._data_in = data_in
    
    def get_ratings_to_dev(self):
        return torch.Tensor(self._data_in[self._idx].toarray()).to(self._device)

def run(model, opts, train_data, batch_size, n_epochs, beta, gamma):
    model.train()
    for epoch in range(n_epochs):
        for batch in generate(batch_size=batch_size, device=device, data_in=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()

            for optimizer in opts:
                optimizer.zero_grad()
                
            _, loss = model(ratings, beta=beta, gamma=gamma)
            loss.backward()
            
            for optimizer in opts:
                optimizer.step()

model_kwargs = {
    'hidden_dim': args.hidden_dim,
    'latent_dim': args.latent_dim,
    'input_dim': train_data.shape[1]
}

model = VAE(**model_kwargs).to(device)  

optimizer_encoder = optim.Adam(model.encoder.parameters(), lr=args.lr)
optimizer_decoder = optim.Adam(model.decoder.parameters(), lr=args.lr)


for epoch in tqdm(range(args.n_epochs)):
    if args.not_alternating:
        run(model=model,
            opts=[optimizer_encoder, optimizer_decoder],
            train_data=train_data,
            batch_size=args.batch_size,
            n_epochs=1,
            beta=args.beta,
            gamma=args.gamma)
    else:
        run(model=model,
            opts=[optimizer_encoder],
            train_data=train_data,
            batch_size=args.batch_size,
            n_epochs=args.n_enc_epochs,
            beta=args.beta,
            gamma=args.gamma)
        
        model.update_prior() 
        
        run(model=model,
            opts=[optimizer_decoder],
            train_data=train_data,
            batch_size=args.batch_size,
            n_epochs=args.n_dec_epochs,
            beta=args.beta,
            gamma=args.gamma)
        
torch.save(model.state_dict(), "./output/vae_model_state.pth")

def recommend_top_k(model, train_data, k=10, batch_size=500):
    """
    Train 데이터 전체에 대해 상위 k개의 추천 아이템을 생성합니다.
    
    Args:
        model: 학습된 VAE 모델
        train_data: 유저-아이템 상호작용 행렬 (CSR 형식)
        k: 추천할 아이템 개수
        batch_size: 배치 크기
    
    Returns:
        recommendations: 각 유저에 대해 k개의 추천 아이템 (리스트 형식)
    """
    model.eval() 
    num_users = train_data.shape[0]
    recommendations = []

    for start_idx in tqdm(range(0, num_users, batch_size), desc="Generating Recommendations"):
        end_idx = min(start_idx + batch_size, num_users)
        batch_users = np.arange(start_idx, end_idx)

        ratings_in = torch.Tensor(train_data[batch_users].toarray()).to(device)

        with torch.no_grad():
            ratings_pred = model(ratings_in, calculate_loss=False).cpu().numpy()

        ratings_pred[train_data[batch_users].nonzero()] = -np.inf

        top_k_items = np.argpartition(-ratings_pred, k, axis=1)[:, :k]
        top_k_sorted = np.argsort(-ratings_pred[np.arange(ratings_pred.shape[0])[:, None], top_k_items], axis=1)
        top_k_items_sorted = top_k_items[np.arange(ratings_pred.shape[0])[:, None], top_k_sorted]

        recommendations.extend(top_k_items_sorted.tolist())

    return recommendations

k = 10  
recommendations = recommend_top_k(model, train_data, k=k)

profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

id2profile = {v: k for k, v in profile2id.items()}

rows = []
for user_idx, movie_idxs in enumerate(recommendations):
    user_id = id2profile[user_idx]  
    for movie_idx in movie_idxs:
        movie_id = unique_sid[movie_idx]  
        rows.append({'user': user_id, 'item': movie_id})

recommendation_df = pd.DataFrame(rows)
recommendation_df = recommendation_df.set_index('user')

print(recommendation_df)

recommendation_df.to_csv("./output/recVAE_submission.csv")
