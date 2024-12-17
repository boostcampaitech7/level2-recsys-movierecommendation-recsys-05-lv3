import os
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader


def Data_Preprocess(config):
    """"
    주어진 학습 데이터("train_ratings.csv")를 읽고, 사용자와 아이템을 재인덱싱한 뒤
    각 사용자의 기록을 학습용과 검증용으로 나누어 반환하는 함수입니다

    Args:
        config (dict): 데이터셋 경로를 포함한 설정 딕셔너리

    Returns:
        tuple: (user_train, user_valid, num_user, num_item)
            user_train (dict): 사용자 인덱스를 키로, 해당 사용자의 학습용 아이템 인덱스 리스트를 값으로 갖는 딕셔너리
            user_valid (dict): 사용자 인덱스를 키로, 해당 사용자의 검증용 아이템 인덱스(마지막 상호작용) 리스트를 값으로 갖는 딕셔너리
            num_user (int): 전체 사용자 수
            num_item (int): 전체 아이템 수
    """
    df = pd.read_csv(os.path.join(config['dataset']['data_path'], "train_ratings.csv"), header=0)
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
    """
    사용자별 상호작용 시퀀스를 바탕으로 BERT 스타일의 마스킹을 적용하여
    학습용 데이터셋을 구성하는 PyTorch Dataset 클래스입니다.

    Args:
        config (dict): 학습에 필요한 파라미터를 담은 설정 딕셔너리.
        user_train (dict): 사용자 인덱스를 키로, 해당 사용자의 아이템 시퀀스를 값으로 갖는 딕셔너리.
        num_user (int): 전체 사용자 수.
        num_item (int): 전체 아이템 수.

    Returns:
        (SeqDataset): PyTorch의 Dataset을 상속한 객체로, __getitem__ 호출 시 (tokens, labels)를 반환합니다.
                    - tokens (torch.LongTensor): 마스킹 및 패딩이 적용된 아이템 시퀀스
                    - labels (torch.LongTensor): 마스킹된 위치에 해당하는 아이템 인덱스(학습 대상), 그 외는 0
    """
    def __init__(self, config, user_train, num_user, num_item):
        self.parameters = config["parameters"]
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = self.parameters["max_len"]
        self.mask_prob = self.parameters["mask_prob"]

    def __len__(self):
        """
        Returns:
            int: 전체 사용자 수(=시퀀스 수)를 반환합니다.
        """
        return self.num_user

    def __getitem__(self, user):
        """
        특정 사용자에 대한 시퀀스를 BERT 스타일 마스킹을 적용한 뒤 반환합니다.

        Args:
            user (int): 사용자 인덱스.

        Returns:
            tuple(torch.LongTensor, torch.LongTensor):
                - tokens: 마스킹 및 패딩 처리가 완료된 아이템 인덱스 시퀀스.
                - labels: 마스킹된 위치의 실제 아이템 인덱스 (학습 대상), 나머지는 0.
        """
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
