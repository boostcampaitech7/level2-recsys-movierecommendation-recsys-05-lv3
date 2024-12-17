import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict


def train(config, model, data_loader):
    """
    주어진 모델과 데이터 로더를 사용하여 학습을 수행하는 함수입니다
    
    Args:
        config (dict): 학습 설정 정보를 담은 딕셔너리
        model (nn.Module): 학습할 PyTorch 모델
        data_loader (DataLoader): 학습에 사용할 데이터로더

    Returns:
        nn.Module: 학습이 완료된 모델 객체
    """
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
    """
    주어진 범위 [l, r) 내에서 집합 s에 없는 랜덤한 정수를 샘플링하는 함수입니다

    Args:
        l (int): 랜덤 정수 샘플링 시작 범위
        r (int): 랜덤 정수 샘플링 종료 범위(미포함)
        s (set): 샘플링 시 제외해야 하는 정수들의 집합

    Returns:
        int: s에 속하지 않는 [l, r) 범위 내의 랜덤한 정수
    """
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def model_eval(config, model, user_train, user_valid, num_user, num_item):
    """
    주어진 모델을 평가하는 함수입니다
    무작위로 선택한 사용자 샘플에 대해 NDCG@10 및 HIT@10을 측정합니다

    Args:
        config (dict): 모델 및 평가에 필요한 설정 정보
        model (nn.Module): 평가할 모델
        user_train (dict): 사용자별 학습용 아이템 시퀀스를 담은 딕셔너리
                            키: 사용자 인덱스, 값: 아이템 인덱스 리스트
        user_valid (dict): 사용자별 검증용 아이템(마지막 상호작용)을 담은 딕셔너리
                            키: 사용자 인덱스, 값: 아이템 인덱스 리스트(검증용 1개 아이템 포함)
        num_user (int): 전체 사용자 수
        num_item (int): 전체 아이템 수

    Returns:
        None: 함수 내에서 NDCG@10, HIT@10 결과를 콘솔에 출력합니다
    """
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