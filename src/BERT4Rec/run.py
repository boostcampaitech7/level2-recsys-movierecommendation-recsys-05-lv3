import argparse

import torch

from .preprocessing import DataLoader, SeqDataset, Data_Preprocess
from .model import BERT4Rec
from .train import train, model_eval
from .inference import inference


def main(args):
    """
    전체 파이프라인을 실행하는 메인 함수입니다
    Args:
        args (argparse.Namespace): 명령행 인자를 파싱한 결과를 담는 객체
            - model_args (dict): 모델 학습에 필요한 파라미터 딕셔너리 (num_epochs, hidden_units, 등)
            - seed (int): 랜덤 시드 설정 값
            - device (torch.device): 모델 및 텐서 연산을 수행할 디바이스(CPU 또는 GPU)
            - model (str): 모델 이름 식별자
            - dataset (dict): 데이터셋 경로, 전처리 경로, 출력 경로 등의 정보를 담은 딕셔너리

    Returns:
        None: 함수 실행 후 모델 학습, 평가, 추론 결과가 해당 경로에 저장됩니다
    """
    config = {}
    config["parameters"] = args.model_args
    config['seed'] = args.seed
    config['device'] = args.device
    config['model'] = args.model
    config['dataset'] = args.dataset
    

    ##### Data Preprocess
    print("##### Data Preprocess ...")
    user_train, user_valid, num_user, num_item = Data_Preprocess(config)

    ##### Load DataLoader
    print("##### Load DataLoader ...")
    seq_dataset = SeqDataset(config, user_train, num_user, num_item)
    data_loader = DataLoader(seq_dataset, shuffle=True, pin_memory=True)

    ##### Load Model
    print("##### Load Model ...")
    model = BERT4Rec(config, num_user, num_item)

    ##### Training
    train(config, model, data_loader)

    ##### Evaluation
    model_eval(config, model, user_train, user_valid, num_user, num_item)

    ##### Inference
    inference(config, model)
