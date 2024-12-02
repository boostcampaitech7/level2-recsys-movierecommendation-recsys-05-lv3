import os
import pandas as pd


DATA_DIR = '../../../data/train'  
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'train_ratings.csv'), header=0)

user_activity = raw_data[['user']].groupby('user', as_index=False).size()
item_popularity = raw_data[['item']].groupby('item', as_index=False).size()

train_plays = raw_data    

unique_sid = pd.unique(train_plays['item'])
unique_uid = pd.unique(train_plays['user'])
show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

pro_dir = os.path.join('./output/')
if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)
with open(os.path.join(pro_dir, 'unique_uid.txt'), 'w') as f:
    for uid in unique_uid:
        f.write('%s\n' % uid)

def numerize(tp):
    uid = map(lambda x: profile2id[x], tp['user'])
    sid = map(lambda x: show2id[x], tp['item'])
    return pd.DataFrame(data={'uid': list(uid), 'sid': list(sid)}, columns=['uid', 'sid'])

train_data = numerize(train_plays)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

