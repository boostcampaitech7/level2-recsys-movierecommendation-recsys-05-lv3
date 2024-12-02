import numpy as np
import torch
from torch import optim
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import argparse

from dataset import *
from model import *
from inference import recommend_top_k  # Import the function

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

# Model setup and training code remains unchanged...

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

# Use the imported function for recommendations
k = 10  
recommendations = recommend_top_k(model, train_data, device=device, k=k)

profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
id2profile = {v: k for k, v in profile2id.items()}

rows = []
for user_idx, movie_idxs in enumerate(recommendations):
    user_id = id2profile[user_idx]  
    for movie_idx in movie_idxs:
        movie_id = unique_sid[movie_idx]  
        rows.append({'user': user_id, 'item': movie_id})

recommendation_df = pd.DataFrame(rows)
recommendation_df.set_index('user', inplace=True)

print(recommendation_df)

recommendation_df.to_csv("../../saved/recVAE_submission.csv")