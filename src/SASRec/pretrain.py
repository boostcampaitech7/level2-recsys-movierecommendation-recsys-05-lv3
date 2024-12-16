import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .datasets import PretrainDataset
from .models import S3RecModel
from .trainers import PretrainTrainer
from .utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs_long,
    set_seed,
)


def run_pretrian(config):

    args = argparse.Namespace()
    args.data_dir = config.dataset.data_path
    args.output_dir = config.dataset.output_path
    args.data_name = config.model_args.data_name
    args.preprocessing_path = config.dataset.preprocessing_path + 'SAS/'

    # model args
    args.model_name = config.model_args.model_name
    args.hidden_size = config.model_args.hidden_size
    args.num_hidden_layers = config.model_args.num_hidden_layers
    args.num_attention_heads = config.model_args.num_attention_heads
    args.hidden_act = config.model_args.hidden_act
    args.attention_probs_dropout_prob = config.model_args.attention_probs_dropout_prob
    args.hidden_dropout_prob = config.model_args.hidden_dropout_prob
    args.initializer_range = config.model_args.initializer_range
    args.max_seq_length = config.model_args.max_seq_length

    args.adam_beta1 = config.model_args.adam_beta1
    args.adam_beta2 = config.model_args.adam_beta2
    args.lr = config.model_args.lr
    args.batch_size = config.model_args.batch_size
    args.weight_decay = config.model_args.weight_decay

    args.pre_epochs = config.model_args.pre_epochs
    args.pre_batch_size = config.model_args.pre_batch_size
    args.mask_p = config.model_args.mask_p
    args.aap_weight = config.model_args.aap_weight
    args.mip_weight = config.model_args.mip_weight
    args.map_weight = config.model_args.map_weight
    args.sp_weight = config.model_args.sp_weight
    args.gpu_id = config.model_args.gpu_id
    args.seed = config.model_args.seed
    
    set_seed(args.seed)
    check_path(args.output_dir)

    args.checkpoint_path = os.path.join(args.preprocessing_path, "Pretrain.pt")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # args.data_file = args.data_dir + args.data_name + '.txt'
    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.preprocessing_path + args.data_name + "_item2attributes.json"
    # concat all user_seq get a long sequence, from which sample neg segment for SP
    user_seq, max_item, long_sequence = get_user_seqs_long(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    args.item2attribute = item2attribute

    model = S3RecModel(args=args)
    trainer = PretrainTrainer(model, None, None, None, None, args)

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)

    for epoch in range(args.pre_epochs):

        pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(
            pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size
        )

        losses = trainer.pretrain(epoch, pretrain_dataloader)

        ## comparing `sp_loss_avg``
        early_stopping(np.array([-losses["sp_loss_avg"]]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
