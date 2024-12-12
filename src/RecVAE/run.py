import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import pandas as pd

from .dataset import *
from .model import *
from .inference import recommend_top_k  
from .preprocessing import *


class Batch:
    def __init__(self, device, idx, data_in):
        self._device = device
        self._idx = idx
        self._data_in = data_in
    
    def get_ratings_to_dev(self):
        return torch.Tensor(self._data_in[self._idx].toarray()).to(self._device)
    
def _generate(batch_size, device, data_in, shuffle=False):
    total_samples = data_in.shape[0]
    idxlist = np.arange(total_samples)

    if shuffle:
        np.random.shuffle(idxlist)

    for st_idx in range(0, total_samples, batch_size):
        end_idx = min(st_idx + batch_size, total_samples)
        idx = idxlist[st_idx:end_idx]
        yield Batch(device, idx, data_in)

def run(model, opts, train_data, batch_size, n_epochs, beta, gamma, device):
    model.train()
    for epoch in range(n_epochs):
        for batch in _generate(batch_size=batch_size, device=device, data_in=train_data, shuffle=True):
            ratings = batch.get_ratings_to_dev()

            for optimizer in opts:
                optimizer.zero_grad()
                
            _, loss = model(ratings, beta=beta, gamma=gamma)
            loss.backward()
            
            for optimizer in opts:
                optimizer.step()

def main(args):

    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Prepare Data-----------------------------------")
    args.dataset.preprocessing_path = args.dataset.preprocessing_path +f'{args.model}'
    if args.EASER.create :
        preprocessor = Preprocessing(data_dir=args.dataset.data_path, output_dir=args.dataset.preprocessing_path, 
                                     threshold=args.RecVAE.threshold, min_items_per_user=args.RecVAE.min_items_per_user, 
                                     min_users_per_item=args.RecVAE.min_users_per_item)
        preprocessor.load_data()
        preprocessor.process()

    loader = DatasetLoader(args.dataset.preprocessing_path, global_indexing=False)
    train_data, unique_sid, unique_uid = loader.get_data()

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
                gamma=gamma,
                device=device)
        else:
            run(model=model,
                opts=[optimizer_encoder],
                train_data=train_data,
                batch_size=batch_size,
                n_epochs=args.RecVAE.n_enc_epochs,
                beta=beta,
                gamma=gamma,
                device=device)
            
            model.update_prior() 
            
            run(model=model,
                opts=[optimizer_decoder],
                train_data=train_data,
                batch_size=batch_size,
                n_epochs=args.RecVAE.n_dec_epochs,
                beta=beta,
                gamma=gamma,
                device=device)

    # 모델 저장하시려면 이 밑의 코드의 주석을 풀어주세요!!!!!!!!
    # torch.save(model.state_dict(), "./output/vae_model_state.pth")

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

    recommendation_df.to_csv(args.dataset.output_path+"recVAE_submission.csv")

if __name__ == '__main__':
    args = parse_args()
    main(args)