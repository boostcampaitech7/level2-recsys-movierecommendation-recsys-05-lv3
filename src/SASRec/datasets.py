import random

import torch
from torch.utils.data import Dataset

from .utils import neg_sample

class PretrainDataset(Dataset):
    """
    Pretrain 모델을 위한 데이터셋 클래스.
    이 클래스는 사용자 상호작용 데이터를 처리하고, 마스킹된 아이템 예측 및 세그먼트 예측을 위한 입력을 준비합니다.

    속성:
    -------
    args : Namespace
        데이터셋을 위한 설정 매개변수(예: max_seq_length, item_size, mask_p, mask_id 등)를 포함하는 인수 객체.
    user_seq : list of list
        각 시퀀스가 아이템 ID의 리스트인 사용자 상호작용 시퀀스의 리스트.
    long_sequence : list
        긴 시퀀스로, 세그먼트 예측에 사용되는 데이터를 포함합니다.
    max_len : int
        입력 시퀀스의 패딩 또는 잘라내기 위해 사용하는 최대 시퀀스 길이.
    part_sequence : list
        훈련에 사용될 부분 시퀀스들을 저장하는 리스트.

    메서드:
    --------
    split_sequence():
        사용자 시퀀스를 일정 크기로 나누어 훈련에 사용할 부분 시퀀스를 준비합니다.
    
    __len__():
        데이터셋의 시퀀스 수를 반환합니다.

    __getitem__(index):
        주어진 인덱스에 대해 마스킹된 아이템 예측, 세그먼트 예측을 위한 데이터 샘플을 반환합니다.
    """
    def __init__(self, args, user_seq, long_sequence):
        """
        제공된 인수, 사용자 시퀀스 및 긴 시퀀스 데이터를 사용하여 PretrainDataset을 초기화합니다.

        매개변수:
        -----------
        args : Namespace
            max_seq_length, item_size, mask_p, mask_id 등 설정을 포함하는 구성 인수 객체.
        user_seq : list of list
            사용자 상호작용 시퀀스, 각 시퀀스는 아이템 ID의 리스트입니다.
        long_sequence : list
            세그먼트 예측에 사용되는 긴 시퀀스 데이터.
        """
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length
        self.part_sequence = []
        self.split_sequence()

    def split_sequence(self):
        """
        사용자 시퀀스를 일정 크기로 나누어 부분 시퀀스를 준비합니다. 
        각 부분 시퀀스는 훈련에 사용됩니다.
        """
        for seq in self.user_seq:
            input_ids = seq[-(self.max_len + 2) : -2]  # 훈련 세트와 동일하게 처리
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[: i + 1])

    def __len__(self):
        """
        데이터셋의 시퀀스 수를 반환합니다.
        
        반환:
        --------
        int
            데이터셋에 있는 부분 시퀀스의 개수.
        """
        return len(self.part_sequence)

    def __getitem__(self, index):
        """
        주어진 인덱스에 대해 마스킹된 아이템 예측, 세그먼트 예측을 위한 데이터 샘플을 반환합니다.
        또한, 아이템과 세그먼트에 대해 마스킹된 값을 처리하고, 관련된 예측 값들을 준비합니다.

        매개변수:
        -----------
        index : int
            가져올 데이터 샘플의 인덱스.

        반환:
        --------
        tuple
            (attributes, masked_item_sequence, pos_items, neg_items, masked_segment_sequence, pos_segment, neg_segment) 형태의 튜플을 반환합니다.
            각 항목은 텐서로 반환되며, 모델의 입력으로 사용됩니다.
        """
        sequence = self.part_sequence[index]  # pos_items
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.args.mask_p:
                masked_item_sequence.append(self.args.mask_id)
                neg_items.append(neg_sample(item_set, self.args.item_size))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # 마지막 항목에 마스크 추가
        masked_item_sequence.append(self.args.mask_id)
        neg_items.append(neg_sample(item_set, self.args.item_size))

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id : start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id : neg_start_id + sample_length]
            masked_segment_sequence = (
                sequence[:start_id]
                + [self.args.mask_id] * sample_length
                + sequence[start_id + sample_length :]
            )
            pos_segment = (
                [self.args.mask_id] * start_id
                + pos_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )
            neg_segment = (
                [self.args.mask_id] * start_id
                + neg_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # 패딩 추가
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0] * pad_len + masked_segment_sequence
        pos_segment = [0] * pad_len + pos_segment
        neg_segment = [0] * pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_len :]
        pos_items = pos_items[-self.max_len :]
        neg_items = neg_items[-self.max_len :]

        masked_segment_sequence = masked_segment_sequence[-self.max_len :]
        pos_segment = pos_segment[-self.max_len :]
        neg_segment = neg_segment[-self.max_len :]

        # 속성 예측
        attributes = []
        for item in pos_items:
            attribute = [0] * self.args.attribute_size
            try:
                now_attribute = self.args.item2attribute[str(item)]
                for a in now_attribute:
                    attribute[a] = 1
            except:
                pass
            attributes.append(attribute)

        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        cur_tensors = (
            torch.tensor(attributes, dtype=torch.long),
            torch.tensor(masked_item_sequence, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
            torch.tensor(masked_segment_sequence, dtype=torch.long),
            torch.tensor(pos_segment, dtype=torch.long),
            torch.tensor(neg_segment, dtype=torch.long),
        )
        return cur_tensors


class SASRecDataset(Dataset):
    """
    SASRec(Self-Attention 기반 순차적 추천) 모델을 위한 데이터셋 클래스.
    이 클래스는 사용자 상호작용 데이터를 처리하고 훈련, 검증, 테스트 또는 제출을 위한 입력을 준비합니다.

    속성:
    -------
    args : Namespace
        데이터셋을 위한 설정 매개변수(예: max_seq_length, item_size 등)를 포함하는 인수 객체.
    user_seq : list of list
        각 시퀀스가 아이템 ID의 리스트인 사용자 상호작용 시퀀스의 리스트.
    test_neg_items : list 또는 None, 선택사항
        테스트를 위한 부정 샘플 아이템 리스트. 기본값은 None.
    data_type : str
        데이터의 유형으로, {'train', 'valid', 'test', 'submission'} 중 하나입니다.
    max_len : int
        입력 시퀀스의 패딩 또는 잘라내기 위해 사용하는 최대 시퀀스 길이.

    메서드:
    --------
    __getitem__(index):
        주어진 인덱스에 대해 훈련, 검증, 테스트 또는 제출을 위한 데이터 샘플을 반환합니다.
    
    __len__():
        데이터셋의 사용자 시퀀스 수를 반환합니다.
    """

    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        """
        제공된 인수, 사용자 시퀀스 및 기타 매개변수로 SASRecDataset을 초기화합니다.

        매개변수:
        -----------
        args : Namespace
            max_seq_length, item_size 등 설정을 포함하는 구성 인수 객체.
        user_seq : list of list
            아이템 ID의 리스트로 구성된 사용자 상호작용 시퀀스의 리스트.
        test_neg_items : list 또는 None, 선택사항
            테스트를 위한 부정 샘플 아이템 리스트, 기본값은 None.
        data_type : str, 선택사항
            데이터 유형으로, {'train', 'valid', 'test', 'submission'} 중 하나여야 하며, 기본값은 'train'.
        """
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):
        """
        주어진 인덱스에 대해 데이터 샘플을 가져옵니다. 
        입력 ID, 양의 타겟 및 부정 타겟을 포함하고, 데이터 유형에 따라 정답을 설정합니다.
        데이터 유형이 'train', 'valid', 'test', 'submission'에 따라 다르게 처리됩니다.

        매개변수:
        -----------
        index : int
            가져올 데이터 샘플의 인덱스.

        반환:
        --------
        tuple
            (user_id, input_ids, target_pos, target_neg, answer, test_samples) 형태의 튜플을 반환합니다. 
            테스트 시 `test_neg_items`가 제공되면 `test_samples`가 포함됩니다.
        """
        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test", "submission"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]

        # submission [0, 1, 2, 3, 4, 5, 6]
        # answer None

        # 데이터 유형에 따라 입력 시퀀스 및 타겟 시퀀스를 준비합니다.
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # 사용되지 않음

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        elif self.data_type == "test":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        else:
            input_ids = items[:]
            target_pos = items[:]  # 사용되지 않음
            answer = []

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        # 시퀀스 길이가 max_len보다 짧을 경우 패딩을 추가하고, 길이를 자릅니다.
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        # 테스트 시 부정 샘플이 있는 경우, 해당 샘플을 추가로 반환합니다.
        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # 테스트를 위한 user_id
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # 테스트를 위한 user_id
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        """
        데이터셋의 길이, 즉 사용자 시퀀스의 개수를 반환합니다.
        """
        return len(self.user_seq)