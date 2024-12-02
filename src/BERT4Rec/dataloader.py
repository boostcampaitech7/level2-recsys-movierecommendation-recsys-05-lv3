import os
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch.utils.data import Dataset


def Data_Preprocess(config):
    parameters = config["parameters"]
    df = pd.read_csv(os.path.join(config["data_path"], "train_ratings.csv"), header=0)

    item_ids = df["item"].unique()
    user_ids = df["user"].unique()
    num_item, num_user = len(item_ids), len(user_ids)

    # user, item indexing
    item2idx = pd.Series(
        data=np.arange(len(item_ids)) + 1, index=item_ids
    )  # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(
        data=np.arange(len(user_ids)), index=user_ids
    )  # user re-indexing (0~num_user-1)

    # dataframe indexing
    df = pd.merge(
        df,
        pd.DataFrame({"item": item_ids, "item_idx": item2idx[item_ids].values}),
        on="item",
        how="inner",
    )
    df = pd.merge(
        df,
        pd.DataFrame({"user": user_ids, "user_idx": user2idx[user_ids].values}),
        on="user",
        how="inner",
    )
    df.sort_values(["user_idx", "time"], inplace=True)
    del df["item"], df["user"]

    # train set, valid set 생성
    users = defaultdict(
        list
    )  # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
    user_train = {}
    user_valid = {}
    for u, i, t in zip(df["user_idx"], df["item_idx"], df["time"]):
        users[u].append(i)

    for user in users:
        user_train[user] = users[user][:-1]
        user_valid[user] = [users[user][-1]]

    print(f"num users: {num_user}, num items: {num_item}")

    return user_train, user_valid, num_user, num_item


class SeqDataset(Dataset):
    def __init__(self, config, user_train, num_user, num_item):
        self.parameters = config["parameters"]
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = self.parameters["max_len"]
        self.mask_prob = self.parameters["mask_prob"]

    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user):
        # iterator를 구동할 때 사용됩니다.
        seq = self.user_train[user]
        tokens = []
        labels = []
        for s in seq:
            prob = (
                np.random.random()
            )  
            if prob < self.mask_prob:
                prob /= self.mask_prob

                # BERT 학습
                if prob < 0.8:
                    # masking
                    tokens.append(
                        self.num_item + 1
                    )  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    tokens.append(
                        np.random.randint(1, self.num_item + 1)
                    )  # item random sampling
                else:
                    tokens.append(s)
                labels.append(s)  # 학습에 사용
            else:
                tokens.append(s)
                labels.append(0)  # 학습에 사용 X, trivial
        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]
        mask_len = self.max_len - len(tokens)

        # zero padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        return torch.LongTensor(tokens), torch.LongTensor(labels)
