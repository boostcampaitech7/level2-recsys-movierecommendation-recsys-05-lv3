import os
import sys

import numpy as np
from scipy import sparse
import pandas as pd

import argparse

sys.argv = sys.argv[:1]
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="../data/train/train_ratings.csv")
parser.add_argument('--output_dir', type=str, default="./output")
parser.add_argument('--threshold', type=float)
parser.add_argument('--min_items_per_user', type=int, default=1)
parser.add_argument('--min_users_per_item', type=int, default=0)

args = parser.parse_args()

dataset = args.dataset
output_dir = args.output_dir
min_uc = args.min_items_per_user
min_sc = args.min_users_per_item

# 데이터 로드 및 전처리
raw_data = pd.read_csv(dataset, header=0)

def get_count(tp, id):
    playcount_groupbyid = tp.groupby(id).size()
    return playcount_groupbyid

def filter_triplets(tp, min_uc=min_uc, min_sc=min_sc): 
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount.index[itemcount >= min_sc])]
    
    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount.index[usercount >= min_uc])]
    
    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item') 
    return tp, usercount, itemcount

raw_data, user_activity, item_popularity = filter_triplets(raw_data)

sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

print("After filtering, there are %d interactions from %d users and %d movies (sparsity: %.3f%%)" % 
      (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

unique_uid = user_activity.index

# 사용자 및 아이템 매핑 생성
unique_sid = pd.unique(raw_data['item'])
show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)
        
with open(os.path.join(output_dir, 'unique_uid.txt'), 'w') as f:
    for uid in unique_uid:
        f.write('%s\n' % uid)

# 학습 데이터 변환 및 저장
def numerize(tp):
    uid = list(map(lambda x: profile2id[x], tp['user']))
    sid = list(map(lambda x: show2id[x], tp['item']))
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

train_data = numerize(raw_data)
train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)