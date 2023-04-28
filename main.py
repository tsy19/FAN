import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import load_data, load_model, process_data
from src.models import MLP
from src.train import train_model

import numpy as np
import os
import time
from src.options import args_parser
from src.IP import IP

def load_info(args):
    if args.dataset == "adult":
        input_dim = 108
        args.optimal_classifier_info = {
            "dims": [input_dim, 300, 300],
            "dropout": 0.5
        }
        args.abstain_classifier_info = {
            "dims": [input_dim + 1, 300, 300],
            "dropout": 0.5
        }
        args.h_classifier_info = {
            "dims": [input_dim + 1, 300, 300],
            "dropout": 0.5
        }
        if args.attribute == "sex":
            args.group1_index = 65
            args.group2_index = 64

    output_names = [
        args.dataset,
        args.attribute,
        args.fairness_notion,
        args.epsilon,
        args.delta1,
        args.delta2,
        args.sigma1,
        args.sigma0,
        args.sigma,
        args.eta1,
        args.eta2,
        args.sample_ratio,
        args.stopping_ratio,
        args.lr,
        args.batch_size,
        args.seed
    ]
    output_names = [str(item) if type(item) != str else item for item in output_names]
    args.output_path = os.path.join(
        ROOT, "result", "TwoGroups",
        "_".join(output_names)
    )

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.optimal_classifier_path = os.path.join(
        ROOT, "result", "OptimalClassifier",
        args.dataset
    )

    args.abstain_classifier_path = os.path.join(
        args.output_path,
        "AbstainClassifier",
    )
    args.h_classifier_path = os.path.join(
        args.output_path,
        "hClassifier",
    )

    return args



if __name__ == '__main__':
    ROOT = "./"

    start_time = time.time()

    args = args_parser()
    print(args)

    args.data_path = os.path.join(
        ROOT, "data", args.dataset
    )

    train_data, test_data = load_data(args)
    print("successfully load data.")

    #load classifiers dimension
    args = load_info(args)

    #check if optimal classifier exists, otherwise train it; load model.
    OptimalNet = MLP(args.optimal_classifier_info)


    if not os.path.isfile(
            os.path.join(args.optimal_classifier_path, "model_state.pth")
    ):
        train_model(train_data, OptimalNet, args, args.optimal_classifier_path)
    OptimalNet = load_model(OptimalNet, args.optimal_classifier_path)


    #integer programming
    num_data = len(train_data)
    sampled_data = train_data[:int(num_data * args.sample_ratio)]
    wn, hn = IP(sampled_data, args, OptimalNet)

    if wn is not None:
        #save hn, wn
        np.savez(os.path.join(
            args.output_path, "IP_results.npz"
            ), w=wn, h=hn
        )
        # train abstain classifier and h
        train_data_w, train_data_h = process_data(sampled_data, OptimalNet, wn, hn)
        AbstainNet = MLP(args.abstain_classifier_info)
        hNet = MLP(args.h_classifier_info)
        train_model(train_data_w, AbstainNet, args, args.abstain_classifier_path)
        train_model(train_data_h, hNet, args, args.h_classifier_path)

    duration = time.time() - start_time
    print('Total Time: {}'.format(duration))