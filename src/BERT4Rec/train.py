import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict


def train(config, model, data_loader):
    parameters = config["parameters"]

    best_n100 = -np.inf

    model.to(config["device"])
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # label이 0인 경우 무시
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, parameters["num_epochs"] + 1):
        tbar = tqdm(data_loader)
        for step, (log_seqs, labels) in enumerate(tbar):
            logits = model(log_seqs)

            # size matching
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1).to(config["device"])

            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tbar.set_description(
                f"Epoch: {epoch:3d}| Step: {step:3d}| Train loss: {loss:.5f}"
            )

        # Save the model.
        with open(
            os.path.join(
                config["dataset"]['preprocessing_path'],
                f"{config['model']}.pt",
            ),
            "wb",
        ) as f:
            torch.save(model, f)

    return model


def random_neg(l, r, s):
    # log에 존재하는 아이템과 겹치지 않도록 sampling
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def model_eval(config, model, user_train, user_valid, num_user, num_item):
    parameters = config["parameters"]

    NDCG = 0.0  # NDCG@10
    HIT = 0.0  # HIT@10

    num_item_sample = 100
    num_user_sample = 1000
    users = np.random.randint(
        0, num_user, num_user_sample
    )  # 1000개만 sampling 하여 evaluation

    model.eval()
    for u in users:
        seq = (user_train[u] + [num_item + 1])[
            -parameters["max_len"] :
        ]  
        rated = set(user_train[u] + user_valid[u])
        item_idx = [user_valid[u][0]] + [
            random_neg(1, num_item + 1, rated) for _ in range(num_item_sample)
        ]

        with torch.no_grad():
            predictions = -model(np.array([seq]))
            predictions = predictions[0][-1][item_idx]  # sampling
            rank = predictions.argsort().argsort()[0].item()

        if rank < 10:  # @10
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1
    print(f"NDCG@10: {NDCG/num_user_sample:.3f}| HIT@10: {HIT/num_user_sample:.3f}")