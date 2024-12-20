import os
import numpy as np
import pandas as pd
from scipy import sparse
from copy import deepcopy
import bottleneck as bn

from .preprocessing import *
from .model import *



def main(args):
    print("Prepare Data-----------------------------------")
    args.dataset.preprocessing_path = args.dataset.preprocessing_path +f'/{args.model}'
    
    if args.EASER.create :
        Preprocessing(args.dataset.data_path,args.dataset.preprocessing_path)

    unique_sid = list()
    with open(os.path.join(f'{args.dataset.preprocessing_path}/unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    unique_uid = list()
    with open(os.path.join(f'{args.dataset.preprocessing_path}/unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())

    train = pd.read_csv(f'{args.dataset.preprocessing_path}/train.csv')
    rows, cols = train['uid'], train['sid']
    X = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=((train['uid'].max() + 1), len(unique_sid)))

    XtX = np.array((X.transpose() * X).todense())
    XtXdiag = deepcopy(np.diag(XtX))

    XtX[np.diag_indices(XtX.shape[0])] = XtXdiag  
    ii_feature_pairs = create_list_feature_pairs(XtX, args.model_args.threshold)
    print("number of feature-pairs: {}".format(len(ii_feature_pairs[0])))

    Z, CCmask = create_matrix_Z(ii_feature_pairs, X)

    ZtZ = np.array((Z.transpose() * Z).todense())
    ZtX = np.array((Z.transpose() * X).todense())
    ZtZdiag = deepcopy(np.diag(ZtZ))


    print("Lets Train-----------------------------------")
    BB, CC = train_higher(XtX, XtXdiag, args.model_args.lambdaBB, ZtZ, ZtZdiag, 
                          args.model_args.lambdaCC, CCmask, ZtX, args.model_args.rho, args.model_args.epochs)
    
    print("Saving model...")
    np.save(f'{args.dataset.preprocessing_path}/BB.npy', BB)
    np.save(f'{args.dataset.preprocessing_path}/CC.npy', CC)
    print("Model saved successfully!")


    print("Lets Inference-----------------------------------")
    pred_val = (X).dot(BB) + Z.dot(CC)
    pred_val[X.nonzero()] = -np.inf  
    
    idx_topk_part = bn.argpartition(-pred_val,  args.model_args.k, axis=1)
    topk_part = pred_val[np.arange(pred_val.shape[0])[:, np.newaxis], idx_topk_part[:, :args.model_args.k]]
    idx_part = np.argsort(-topk_part, axis=1)
    top_k_recommendations = idx_topk_part[np.arange(pred_val.shape[0])[:, np.newaxis], idx_part]

    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))
    id2profile = {v: k for k, v in profile2id.items()}

    rows = []
    for user_idx, movie_idxs in enumerate(top_k_recommendations):
        user_id = id2profile[user_idx]  
        for movie_idx in movie_idxs:
            movie_id = unique_sid[movie_idx]  
            rows.append({'user': user_id, 'item': movie_id})

    recommendation_df = pd.DataFrame(rows)
    recommendation_df = recommendation_df.set_index('user')

    print(recommendation_df)

    recommendation_df.to_csv(f'{args.dataset.output_path}/EASER.csv')

