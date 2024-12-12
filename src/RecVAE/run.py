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
from .preprocessing import *


def main(args):

    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Prepare Data-----------------------------------")
    args.dataset.preprocessing_path = args.dataset.preprocessing_path +f'/{args.model}'
    if args.EASER.create :
        preprocessor = Preprocessing(data_dir=args.dataset.data_path, output_dir=args.dataset.preprocessing_path, 
                                     threshold=args.RecVAE.threshold, min_items_per_user=args.RecVAE.min_items_per_user, 
                                     min_users_per_item=args.RecVAE.min_users_per_item)
        preprocessor.load_data()
        preprocessor.process()




    train_data, unique_sid, unique_uid = get_data(args.dataset.preprocessing_path)  

    batch_size= args.RecVAE.batch_size
    beta= args.RecVAE.beta
    gamma= args.RecVAE.gamma
    lr=args.RecVAE.lr
    n_epochs= args.RecVAE.n_epochs

    model_kwargs = {
        'hidden_dim': args.RecVAE.hidden_dim,
        'latent_dim': args.RecVAE.latent_dim,
        'input_dim': train_data.shape[1]
    }

    model = VAE(**model_kwargs).to(device)  

    optimizer_encoder = optim.Adam(model.encoder.parameters(), lr=lr)
    optimizer_decoder = optim.Adam(model.decoder.parameters(), lr=lr)

    for epoch in tqdm(range(n_epochs)):
        if args.RecVAE.not_alternating:
            run(model=model,
                opts=[optimizer_encoder, optimizer_decoder],
                train_data=train_data,
                batch_size=batch_size,
                n_epochs=n_epochs,
                beta=beta,
                gamma=gamma)
        else:
            run(model=model,
                opts=[optimizer_encoder],
                train_data=train_data,
                batch_size=batch_size,
                n_epochs=args.RecVAE.n_enc_epochs,
                beta=beta,
                gamma=gamma)
            
            model.update_prior() 
            
            run(model=model,
                opts=[optimizer_decoder],
                train_data=train_data,
                batch_size=batch_size,
                n_epochs=args.RecVAE.n_dec_epochs,
                beta=beta,
                gamma=gamma)

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

if __name__ == '__main__':
    args = parse_args()
    main(args)