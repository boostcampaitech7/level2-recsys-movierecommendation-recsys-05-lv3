import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from .preprocessing import item2attributes
from .datasets import SASRecDataset
from .models import S3RecModel
from .pretrain import run_pretrian
from .trainers import FinetuneTrainer
from .utils import (
    EarlyStopping,
    check_path,
    generate_submission_file,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)


def main(config):

    if not config.model_args.using_pretrain : 
        config.model = 'SAS'
    else :
        config.model = 'S3R'

    if config.model_args.item_attribute_create :
        item2attributes(config)

    if config.model_args.create_pretrain :
        run_pretrian(config)
    
    args = argparse.Namespace()

    # model info
    args.model = config.model 
    args.data_name = config.model_args.data_name

    # model path
    args.data_dir = config.dataset.data_path
    args.preprocessing_path = config.dataset.preprocessing_path + 'SAS/'
    args.output_dir = config.dataset.output_path

    args.seed = config.model_args.seed
    args.log_freq = config.model_args.log_freq
    args.gpu_id = config.model_args.gpu_id
    args.no_cuda = config.model_args.no_cuda 
    args.using_pretrain = config.model_args.using_pretrain

    # model args
    args.model_name = config.model_args.model_name
    args.hidden_size = config.model_args.hidden_size
    args.num_hidden_layers = config.model_args.num_hidden_layers
    args.num_attention_heads = config.model_args.num_attention_heads
    args.hidden_act = config.model_args.hidden_act
    args.attention_probs_dropout_prob = config.model_args.attention_probs_dropout_prob
    args.hidden_dropout_prob = config.model_args.hidden_dropout_prob
    args.initializer_range = config.model_args.initializer_range
    args.max_seq_length = config.model_args.max_seq_lengthhs
    args.weight_decay = config.model_args.weight_decay
    args.adam_beta1 = config.model_args.adam_beta1
    args.adam_beta2 = config.model_args.adam_beta2
    
    # train
    args.lr = config.model_args.lr 
    args.batch_size = config.model_args.batch_size
    args.epochs = config.model_args.epoc

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.preprocessing_path + args.data_name + "_item2attributes.json"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, submission_rating_matrix = get_user_seqs(
        args.data_file
    )

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.preprocessing_path, args_str + ".txt")
    args.item2attribute = item2attribute

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.preprocessing_path, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    test_dataset = SASRecDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.batch_size
    )

    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(
        model, train_dataloader, eval_dataloader, test_dataloader, None, args
    )

    if args.using_pretrain:
        pretrained_path = os.path.join(args.preprocessing_path, "Pretrain.pt")
        try:
            trainer.load(pretrained_path)
            print(f"Load Checkpoint From {pretrained_path}!")

        except FileNotFoundError:
            print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    else:
        print("Not using pretrained model. The Model is same as SASRec")

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("---------------Change to test_rating_matrix!-------------------")
    trainer.args.train_matrix = test_rating_matrix

    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))

    submission_dataset = SASRecDataset(args, user_seq, data_type="submission")
    submission_sampler = SequentialSampler(submission_dataset)
    submission_dataloader = DataLoader(
        submission_dataset, sampler=submission_sampler, batch_size=args.batch_size
    )

    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(model, None, None, None, submission_dataloader, args)

    trainer.load(args.checkpoint_path)

    print(f"Load model from {args.checkpoint_path} for submission!")
    preds = trainer.submission(0)

    generate_submission_file(args, preds)

