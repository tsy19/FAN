import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import load_data, load_model, process_data, sample_data, get_stats
from src.models import MLP
from src.train import train_model
import matplotlib.pyplot as plt

import pickle
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
            args.group_indices = [65, 64]

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
        args.lr,
        args.batch_size,
        args.seed,
        args.max_epoch,
        args.min_epoch,
        args.patience
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

def plot(args):
    with open(os.path.join(args.output_path, 'stats.pickle'), 'rb') as f:
        train_stats_raw = pickle.load(f)
        train_stats = pickle.load(f)
        test_stats = pickle.load(f)
        train_optimal_stats = pickle.load(f)
        test_optimal_stats = pickle.load(f)
    legend = ["Train data (Optimal)", "Test data (Optimal)", "Train data (IP)", "Train data (NN)", "Test data (NN)"]
    stats = [train_optimal_stats, test_optimal_stats, train_stats_raw, train_stats, test_stats]
    plot_stats = [
        "num_of_data",
        "num_of_positive_data",
        "num_of_negative_data",
        "accuracy",
        "abstain_rate",
        "abstain_rate_positive",
        "abstain_rate_negative",
        # "flip_rate",
        # "flip_rate_positive",
        # "flip_rate_negative",
        "accept_rate",
        # "accept_rate_no_abstain",
        "true_positive_rate",
        "true_positive_rate_no_abstain",
        # "false_negative_rate",
        # "false_negative_rate_no_abstain",
        "true_negative_rate",
        "true_negative_rate_no_abstain",
        # "false_positive_rate",
        # "false_positive_rate_no_abstain",
    ]
    COL = 4
    ROW = len(plot_stats) // COL
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(nrows=ROW, ncols=COL, figsize=(COL * 8, ROW * 6))
    for i in range(len(plot_stats)):
        row = i // COL
        col = i - COL * row

        key = plot_stats[i]
        title = key.title().replace("_", " ")
        title = title.replace('No', 'without')

        bar_width = 1 / (len(legend) + 1)
        x_labels = ["Overall", "G1", "G2"]
        x_pos = np.arange(len(x_labels))
        start = x_pos - int(len(legend) / 2) * bar_width
        for j, item in enumerate(stats):
            ax[row,col].bar(start + j * bar_width, item[key], width=bar_width, label=legend[j])
            ax[row,col].set_title(title)
            ax[row,col].set_xticks(x_pos)
            ax[row,col].set_xticklabels(x_labels)
            ax[row,col].legend(loc="upper right", framealpha=0.3)
    fig.savefig(os.path.join(args.output_path, 'plot_stats.pdf'), bbox_inches='tight')
    return


def evaluate(args):
    train_data, _, test_data = load_data(args)
    file_path = os.path.join(args.output_path, "IP_results.npz")
    IP_results = np.load(file_path)
    wn = IP_results['w']
    hn = IP_results['h']
    OptimalNet = MLP(args.optimal_classifier_info)
    AbstainNet = MLP(args.abstain_classifier_info)
    hNet = MLP(args.h_classifier_info)
    OptimalNet = load_model(OptimalNet, args.optimal_classifier_path)
    AbstainNet = load_model(AbstainNet, args.abstain_classifier_path)
    hNet = load_model(hNet, args.h_classifier_path)
    train_optimal_stats, train_stats, train_stats_raw = get_stats(train_data, args, AbstainNet, hNet, OptimalNet=OptimalNet, wn=wn, hn=hn)
    test_optimal_stats, test_stats = get_stats(test_data, args, AbstainNet, hNet, OptimalNet=OptimalNet)

    with open(os.path.join(args.output_path, 'stats.pickle'), 'wb') as f:
        pickle.dump(train_stats_raw, f)
        pickle.dump(train_stats, f)
        pickle.dump(test_stats, f)
        pickle.dump(train_optimal_stats, f)
        pickle.dump(test_optimal_stats, f)

    plot(args)


if __name__ == '__main__':
    ROOT = "./"



    args = args_parser()
    print(args)

    args.data_path = os.path.join(
        ROOT, "data", args.dataset
    )

    #load classifiers dimension
    args = load_info(args)

    if args.evaluate:
        evaluate(args)
    else:
        start_time = time.time()
        train_data, val_data, test_data = load_data(args)
        print("successfully load data.")

        #check if optimal classifier exists, otherwise train it; load model.
        OptimalNet = MLP(args.optimal_classifier_info)

        if not os.path.isfile(
                os.path.join(args.optimal_classifier_path, "model_state.pth")
        ):
            train_model(train_data, val_data, OptimalNet, args, args.optimal_classifier_path)
        OptimalNet = load_model(OptimalNet, args.optimal_classifier_path)


        #integer programming
        num_data = len(train_data)
        sampled_data = sample_data(train_data, int(num_data * args.sample_ratio))
        wn, hn = IP(sampled_data, args, OptimalNet)

        if wn is not None:
            #save hn, wn
            np.savez(os.path.join(
                args.output_path, "IP_results.npz"
                ), w=wn, h=hn
            )
            # train abstain classifier and h
            train_data_w, train_data_h, val_data_w, val_data_h = process_data(sampled_data, args, OptimalNet, True, wn, hn)
            AbstainNet = MLP(args.abstain_classifier_info)
            hNet = MLP(args.h_classifier_info)
            train_model(train_data_w, val_data_w, AbstainNet, args, args.abstain_classifier_path)
            train_model(train_data_h, val_data_h, hNet, args, args.h_classifier_path)

            evaluate(args)
        duration = time.time() - start_time
        print('Total Time: {}'.format(duration))