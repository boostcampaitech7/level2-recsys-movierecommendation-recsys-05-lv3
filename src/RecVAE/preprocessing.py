import os
import pandas as pd

class Preprocessing:
    def __init__(self, data_dir, output_dir, threshold=None, min_items_per_user=1, min_users_per_item=0):
        self.dataset = data_dir+'train_ratings.csv'
        self.output_dir = output_dir
        self.threshold = threshold
        self.min_uc = min_items_per_user
        self.min_sc = min_users_per_item
        self.raw_data = None

    def load_data(self):
        try:
            self.raw_data = pd.read_csv(self.dataset, header=0)
        except FileNotFoundError:
            print(f"Dataset file {self.dataset} not found.")

    def get_count(self, tp, id):
        playcount_groupbyid = tp.groupby(id).size()
        return playcount_groupbyid

    def filter_triplets(self, tp):
        if self.min_sc > 0:
            itemcount = self.get_count(tp, 'item')
            tp = tp[tp['item'].isin(itemcount.index[itemcount >= self.min_sc])]
        
        if self.min_uc > 0:
            usercount = self.get_count(tp, 'user')
            tp = tp[tp['user'].isin(usercount.index[usercount >= self.min_uc])]
        
        usercount, itemcount = self.get_count(tp, 'user'), self.get_count(tp, 'item') 
        return tp, usercount, itemcount

    def process(self):
        if self.raw_data is None:
            print("No data loaded. Please check the dataset path.")
            return

        self.raw_data, user_activity, item_popularity = self.filter_triplets(self.raw_data)

        sparsity = 1. * len(self.raw_data) / (len(user_activity) * len(item_popularity))

        print("After filtering, there are %d interactions from %d users and %d movies (sparsity: %.3f%%)" % 
              (len(self.raw_data), len(user_activity), len(item_popularity), sparsity * 100))

        unique_uid = user_activity.index

        unique_sid = pd.unique(self.raw_data['item'])
        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(os.path.join(self.output_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)
                
        with open(os.path.join(self.output_dir, 'unique_uid.txt'), 'w') as f:
            for uid in unique_uid:
                f.write('%s\n' % uid)

        def numerize(tp):
            uid = list(map(lambda x: profile2id[x], tp['user']))
            sid = list(map(lambda x: show2id[x], tp['item']))
            return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

        train_data = numerize(self.raw_data)
        train_data.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)

