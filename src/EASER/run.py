import os
import numpy as np
import pandas as pd
from scipy import sparse
from copy import deepcopy
import bottleneck as bn
import argparse

from model import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--threshold', type=int, default=3500)
    parser.add_argument('--lambdaBB', type=int, default=1000)
    parser.add_argument('--lambdaCC', type=int, default=10000)
    parser.add_argument('--rho', type=int, default=50000)
    parser.add_argument('--k', type=int, default=10)

    return parser.parse_args()


def main(args):
    print("Prepare Data-----------------------------------")
    unique_sid = list()
    with open(os.path.join('./output/unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    unique_uid = list()
    with open(os.path.join('./output/unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())

    train = pd.read_csv('./output/train.csv')
    rows, cols = train['uid'], train['sid']
    X = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=((train['uid'].max() + 1), len(unique_sid)))

    XtX = np.array((X.transpose() * X).todense())
    XtXdiag = deepcopy(np.diag(XtX))

    XtX[np.diag_indices(XtX.shape[0])] = XtXdiag  
    ii_feature_pairs = create_list_feature_pairs(XtX, args.threshold)
    print("number of feature-pairs: {}".format(len(ii_feature_pairs[0])))

    Z, CCmask = create_matrix_Z(ii_feature_pairs, X)

    ZtZ = np.array((Z.transpose() * Z).todense())
    ZtX = np.array((Z.transpose() * X).todense())
    ZtZdiag = deepcopy(np.diag(ZtZ))


    print("Lets Train-----------------------------------")
    BB, CC = train_higher(XtX, XtXdiag, args.lambdaBB, ZtZ, ZtZdiag, args.lambdaCC, CCmask, ZtX, args.rho, args.epochs)
    
    print("Saving model...")
    np.save('./output/BB.npy', BB)
    np.save('./output/CC.npy', CC)
    print("Model saved successfully!")


    print("Lets Inference-----------------------------------")
    pred_val = (X).dot(BB) + Z.dot(CC)
    pred_val[X.nonzero()] = -np.inf  
    
    idx_topk_part = bn.argpartition(-pred_val,  args.k, axis=1)
    topk_part = pred_val[np.arange(pred_val.shape[0])[:, np.newaxis], idx_topk_part[:, :args.k]]
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

    recommendation_df.to_csv("../../saved/easer.csv")


if __name__ == '__main__':
    args = parse_args()
    main(args)
