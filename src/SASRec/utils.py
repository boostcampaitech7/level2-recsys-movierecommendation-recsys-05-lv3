import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose:
            print(f"Better performance. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    """
    주어진 텐서에서 상위 k개의 값을 선택하는 함수입니다.

    매개변수
    ----------
    x : torch.Tensor
        상위 k개 값을 선택할 입력 텐서
    dim : int
        값을 선택할 차원
    k : int
        선택할 상위 k개의 개수

    반환값
    -------
    torch.Tensor
        선택된 상위 k개의 값이 포함된 텐서
    """
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    """
    주어진 텐서에서 지정된 차원의 평균을 계산하는 함수입니다.

    매개변수
    ----------
    x : torch.Tensor
        평균을 계산할 입력 텐서
    dim : int
        평균을 계산할 차원

    반환값
    -------
    torch.Tensor
        지정된 차원의 평균값
    """
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    """
    검증용 평가 행렬을 생성하는 함수입니다.

    매개변수
    ----------
    user_seq : list of list
        각 사용자의 아이템 시퀀스
    num_users : int
        사용자 수
    num_items : int
        아이템 수

    반환값
    -------
    csr_matrix
        검증용 평가 행렬 (희소 행렬)
    """
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    """
    테스트용 평가 행렬을 생성하는 함수입니다.

    매개변수
    ----------
    user_seq : list of list
        각 사용자의 아이템 시퀀스
    num_users : int
        사용자 수
    num_items : int
        아이템 수

    반환값
    -------
    csr_matrix
        테스트용 평가 행렬 (희소 행렬)
    """
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_submission(user_seq, num_users, num_items):
    """
    제출용 평가 행렬을 생성하는 함수입니다.

    매개변수
    ----------
    user_seq : list of list
        각 사용자의 아이템 시퀀스
    num_users : int
        사용자 수
    num_items : int
        아이템 수

    반환값
    -------
    csr_matrix
        제출용 평가 행렬 (희소 행렬)
    """
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_submission_file(args, preds):
    """
    예측 결과를 파일로 저장하는 함수입니다.

    매개변수
    ----------
    args : Namespace
        인자들을 담고 있는 객체 (예: 출력 디렉토리 경로)
    preds : list of list
        예측된 아이템들

    반환값
    -------
    없음
    """
    rating_df = pd.read_csv(args.data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        f"{args.output_dir}/{args.model}.csv", index=False
    )


def get_user_seqs(data_file):
    """
    데이터 파일에서 사용자별 아이템 시퀀스를 추출하고, 유효, 테스트 및 제출용 평가 행렬을 생성하는 함수입니다.

    매개변수
    ----------
    data_file : str
        사용자-아이템 평가 데이터를 담고 있는 파일 경로

    반환값
    -------
    tuple
        사용자 시퀀스, 최대 아이템 ID, 유효/테스트/제출용 평가 행렬
    """ 
    rating_df = pd.read_csv(data_file)
    lines = rating_df.groupby("user")["item"].apply(list)
    user_seq = []
    item_set = set()
    for line in lines:

        items = line
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    submission_rating_matrix = generate_rating_matrix_submission(
        user_seq, num_users, num_items
    )
    return (
        user_seq,
        max_item,
        valid_rating_matrix,
        test_rating_matrix,
        submission_rating_matrix,
    )


def get_user_seqs_long(data_file):
    """
    데이터 파일에서 사용자별 아이템 시퀀스를 추출하고, 긴 시퀀스와 사용자 시퀀스를 반환하는 함수입니다.

    매개변수
    ----------
    data_file : str
        사용자-아이템 평가 데이터를 담고 있는 파일 경로

    반환값
    -------
    tuple
        사용자 시퀀스, 최대 아이템 ID, 전체 아이템 시퀀스
    """
    rating_df = pd.read_csv(data_file)
    lines = rating_df.groupby("user")["item"].apply(list)
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        items = line
        long_sequence.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence


def get_item2attribute_json(data_file):
    """
    아이템과 그 속성 간의 매핑을 담은 JSON 파일을 읽어들이고, 속성의 크기를 계산하는 함수입니다.

    매개변수
    ----------
    data_file : str
        아이템 속성을 담고 있는 JSON 파일 경로

    반환값
    -------
    tuple
        아이템-속성 매핑, 속성 크기
    """
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    """
    예측 결과에 대해 HIT, NDCG, MRR 등의 평가 지표를 계산하는 함수입니다.

    매개변수
    ----------
    pred_list : list
        예측된 순위 리스트
    topk : int, 선택적
        평가할 상위 k값 (기본값은 10)

    반환값
    -------
    tuple
        HIT, NDCG, MRR 값
    """
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    """
    개별 샘플에 대해 k에서의 정밀도를 계산하는 함수입니다.

    매개변수
    ----------
    actual : list
        실제 아이템 리스트
    predicted : list
        예측된 아이템 리스트
    topk : int
        평가할 상위 k값

    반환값
    -------
    float
        해당 샘플에 대한 정밀도 값
    """
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    """
    전체 사용자에 대해 k에서의 정밀도를 계산하는 함수입니다.

    매개변수
    ----------
    actual : list of list
        각 사용자에 대한 실제 아이템 리스트
    predicted : list of list
        각 사용자에 대한 예측된 아이템 리스트
    topk : int
        평가할 상위 k값

    반환값
    -------
    float
        전체 사용자에 대한 정밀도 값
    """
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    """
    k에서의 재현율을 계산하는 함수입니다.

    매개변수
    ----------
    actual : list of list
        각 사용자에 대한 실제 아이템 리스트
    predicted : list of list
        각 사용자에 대한 예측된 아이템 리스트
    topk : int
        평가할 상위 k값

    반환값
    -------
    float
        전체 사용자에 대한 재현율 값
    """
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    k에서의 평균 정밀도를 계산합니다.
    이 함수는 두 개의 아이템 리스트에 대해 k에서의 평균 정밀도를 계산합니다.
    
    매개변수
    ----------
    actual : list
        예측해야 할 아이템들의 리스트입니다. (리스트 내 순서는 중요하지 않음)
    predicted : list
        예측된 아이템들의 리스트입니다. (리스트 내 순서는 중요)
    k : int, 선택적
        예측된 아이템들의 최대 개수

    반환값
    -------
    score : float
        입력된 리스트에 대한 k에서의 평균 정밀도 값
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    k에서 평균 정밀도(Mean Average Precision, MAP)를 계산합니다.
    이 함수는 두 개의 아이템 리스트의 리스트에 대해 k에서의 평균 정밀도를 계산합니다.
    
    매개변수
    ----------
    actual : list
        예측해야 할 아이템들의 리스트의 리스트 (리스트 내 순서는 중요하지 않음)
    predicted : list
        예측된 아이템들의 리스트의 리스트 (리스트 내 순서는 중요)
    k : int, 선택적
        예측된 아이템들의 최대 개수

    반환값
    -------
    score : float
        입력된 리스트에 대한 k에서의 평균 정밀도 값
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    """
    주어진 실제 값(actual)과 예측 값(predicted)에 대해 NDCG@k (Normalized Discounted Cumulative Gain) 점수를 계산하는 함수입니다.
    NDCG는 추천 시스템에서 추천된 항목들의 순서를 평가하는 지표로, 높은 순위에 더 높은 가중치를 부여하여 평가합니다.

    Args:
        actual (list of list): 각 사용자에 대한 실제 아이템 리스트. 각 사용자마다 실제 아이템의 순서대로 목록이 포함됩니다.
        predicted (list of list): 각 사용자에 대한 예측된 아이템 리스트. 각 사용자마다 예측된 아이템의 순서대로 목록이 포함됩니다.
        topk (int): 계산할 상위 k개 항목에 대해 NDCG를 평가합니다.

    Returns:
        float: 전체 사용자의 평균 NDCG@k 점수.

    """
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))



def idcg_k(k):
    """
    주어진 k에 대해 이상적인 할인 누적 이득(DCG, Discounted Cumulative Gain)을 계산하는 함수입니다.
    이상적인 DCG는 실제 결과의 순위가 가장 좋은 경우에 대한 DCG를 나타냅니다.

    Args:
        k (int): 계산할 상위 k개 항목에 대해 DCG를 평가합니다.

    Returns:
        float: 주어진 k에 대한 이상적인 DCG 값.
    """
    res = su
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
