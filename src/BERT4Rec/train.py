import argparse

import torch

from utils import Setting, load_config
from dataloader import DataLoader, SeqDataset, Data_Preprocess
from model import BERT4Rec
from trainer import run, model_eval


def main(args):
    ##### Load config
    print("##### Load config ...")
    config = load_config(args)
    Setting.seed_everything(config["seed"])
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

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
    run(config, model, data_loader)

    ##### Evaluation
    model_eval(config, model, user_train, user_valid, num_user, num_item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    #### Environment Settings
    arg("--model", "-m", type=str, help="select model")
    arg("--config_ver", "-c", type=str, help="veresion of experiments")

    args = parser.parse_args()
    main(args)
