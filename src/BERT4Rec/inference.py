import os
import argparse

import torch

from utils import Setting, load_config
from trainer import inference


def main(args):
    ##### Load config
    print("##### Load config ...")
    config = load_config(args)
    Setting.seed_everything(config["seed"])
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    ##### Load Model
    print("##### Load Model ...")
    with open(
        os.path.join(
            config["model_save_path"], f"{config['model']}_V_{config['config_ver']}.pt"
        ),
        "rb",
    ) as f:
        model = torch.load(f)

    ##### Inference
    inference(config, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    #### Environment Settings
    arg("--model", "-m", type=str, help="select model")
    arg("--config_ver", "-c", type=str, help="veresion of experiments")

    args = parser.parse_args()
    main(args)