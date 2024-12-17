import argparse

import torch

from .preprocessing import DataLoader, SeqDataset, Data_Preprocess
from .model import BERT4Rec
from .train import train, model_eval
from .inference import inference


def main(args):    
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
