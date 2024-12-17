import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict


def inference(config, model):
    parameters = config["parameters"]

    df = pd.read_csv(os.path.join(config["dataset"]["data_path"], "train_ratings.csv"), header=0)
    item_ids = df["item"].unique()
    user_ids = df["user"].unique()
    num_item, num_user = len(item_ids), len(user_ids)

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

    # 인덱스로 변경된 Raw Data 세팅 (user_df)
    users_dic = defaultdict(list)
    user_df = {}
    for u, i, t in zip(df["user_idx"], df["item_idx"], df["time"]):
        users_dic[u].append(i)

    for user in users_dic:
        user_df[user] = users_dic[user]

    # inference
    model.eval()
    predict_list = []
    for u in tqdm(range(num_user)):
        seq = (user_df[u] + [num_item + 1])[-parameters["max_len"] :]
        used_items_list = [
            a - 1 for a in user_df[u]
        ]  # 사용한 아이템에 대해 인덱스 계산을 위해 1씩 뺀다.

        if len(seq) < parameters["max_len"]:
            seq = np.pad(
                seq,
                (parameters["max_len"] - len(seq), 0),
                "constant",
                constant_values=0,
            )  # 패딩 추가

        with torch.no_grad():
            predictions = -model(np.array([seq]))
            predictions = predictions[0][-1][1:]  # mask 제외
            predictions[used_items_list] = np.inf  # 사용한 아이템은 제외하기 위해 inf
            rank = predictions.argsort().argsort().tolist()

            for i in range(10):
                rank.index(i)
                predict_list.append([u, rank.index(i)])

    # Data Export
    # 인덱스를 원래 데이터 상태로 변환하여 csv 저장합니다.
    predict_list_idx = [
        [user2idx.index[user], item2idx.index[item]] for user, item in predict_list
    ]
    predict_df = pd.DataFrame(data=predict_list_idx, columns=["user", "item"])
    predict_df = predict_df.sort_values("user")
    predict_df.to_csv(
        os.path.join(
            config["dataset"]["output_path"], f"{config['model']}.csv"
        ),
        index=False,
    )

    print("Inference Done!")