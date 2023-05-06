import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import load_data, load_model, process_data, sample_data, get_stats
from src.models import MLP
from src.train import train_model
import matplotlib.pyplot as plt
import shutil

import random
import json
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

    if args.dataset == "compas":
        input_dim = 7
        args.optimal_classifier_info = {
            "dims": [input_dim, 150, 150],
            "dropout": 0.5
        }
        args.abstain_classifier_info = {
            "dims": [input_dim + 1, 150, 150],
            "dropout": 0.5
        }
        args.h_classifier_info = {
            "dims": [input_dim + 1, 150, 150],
            "dropout": 0.5
        }
        if args.attribute == "race":
            args.group_indices = 3
        elif args.attribute == "sex":
            args.group_indices = 4
    if args.dataset == "law":
        input_dim = 11
        args.optimal_classifier_info = {
            "dims": [input_dim, 150, 150],
            "dropout": 0.5
        }
        args.abstain_classifier_info = {
            "dims": [input_dim + 1, 150, 150],
            "dropout": 0.5
        }
        args.h_classifier_info = {
            "dims": [input_dim + 1, 150, 150],
            "dropout": 0.5
        }
        if args.attribute == "race":
            args.group_indices = 9


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
        args.ROOT, "result", "TwoGroups",
        "_".join(output_names)
    )

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    args.optimal_classifier_path = os.path.join(
        args.ROOT, "result", "OptimalClassifier",
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

def set_seed(seed):
    """
    Set the seed for standard libraries and cuda.

    Args:
        seed: The seed number that should ber used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def plot(args):
    def list_sub(list1, list2):
        return [list1[i] - list2[i] for i in range(len(list1))]
    with open(os.path.join(args.output_path, 'stats.pkl'), 'rb') as f:
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


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(2 * 8, 1 * 6))
    x_values = [0, 1, 2, 3, 4]
    x_labels = ["Train data (Optimal)", "Test data (Optimal)", "Train data (IP)", "Train data (NN)", "Test data (NN)"]
    data_fairness = []
    if args.fairness_notion == "DP":
        title = "Disparity (Demographic Parity)"
        name = "accept_rate"
    elif args.fairness_notion == "EO":
        title = "Disparity (Equal Opportunity)"
        name = "true_positive_rate"
    for item in stats:
        data_fairness.append( abs(item[name][0] - item[name][1] ) )
    ax[0].bar(x_labels, data_fairness)
    ax[0].set_xticks(x_values)
    ax[0].set_xticklabels(x_labels, rotation=90)
    ax[0].set_title(title)

    legend = ["Train data (IP)", "Train data (NN)", "Test data (NN)"]
    data_accuracy = [
        list_sub( train_stats_raw["accuracy"], train_optimal_stats["accuracy"]),
        list_sub(train_stats["accuracy"], train_optimal_stats["accuracy"]),
        list_sub(test_stats["accuracy"], test_optimal_stats["accuracy"]),
    ]
    bar_width = 1 / (len(legend) + 1)
    x_values = [0, 1, 2]
    x_labels = ["Overall", "G1", "G2"]
    x_pos = np.arange(len(x_labels))
    start = x_pos - int(len(legend) / 2) * bar_width
    for j, item in enumerate(data_accuracy):
        ax[1].bar(start + j * bar_width, item, width=bar_width, label=legend[j])
    ax[1].set_xticks(x_values)
    ax[1].set_xticklabels(x_labels, rotation=90)
    ax[1].set_title("Increase in Accuracy")
    ax[1].legend(loc="upper right", framealpha=0.3)
    fig.savefig(os.path.join(args.output_path, 'plot_stats_2.pdf'), bbox_inches='tight')

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

    with open(os.path.join(args.output_path, 'stats.pkl'), 'wb') as f:
        pickle.dump(train_stats_raw, f)
        pickle.dump(train_stats, f)
        pickle.dump(test_stats, f)
        pickle.dump(train_optimal_stats, f)
        pickle.dump(test_optimal_stats, f)

    # plot(args)


if __name__ == '__main__':

    args = args_parser()
    print(args)

    ROOT = args.ROOT

    torch.cuda.empty_cache()
    set_seed(args.seed)


    args.data_path = os.path.join(
        ROOT, "data", args.dataset
    )

    #load classifiers dimension
    args = load_info(args)
    args_dict = vars(args)
    json_str = json.dumps(args_dict)
    with open(os.path.join(args.output_path, 'args.json'), 'w') as f:
        f.write(json_str)

    if args.evaluate:
        evaluate(args)
    else:
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
        start_time = time.time()
        wn, hn = IP(sampled_data, args, OptimalNet)
        IP_duration = time.time() - start_time
        if wn is None:
            print("Not feasible!")
        else:
            #save hn, wn
            np.savez(os.path.join(
                args.output_path, "IP_results.npz"
                ), w=wn, h=hn
            )
            # train abstain classifier and h
            train_data_w, train_data_h, val_data_w, val_data_h = process_data(sampled_data, args, OptimalNet, True, wn, hn)

            AbstainNet = MLP(args.abstain_classifier_info)
            hNet = MLP(args.h_classifier_info)

            start_time = time.time()
            train_model(train_data_w, val_data_w, AbstainNet, args, args.abstain_classifier_path)
            Abstain_duration = time.time() - start_time
            start_time = time.time()
            train_model(train_data_h, val_data_h, hNet, args, args.h_classifier_path)
            h_duration = time.time() - start_time

            running_time_dict = {
                "Integer Programming": IP_duration,
                "Abstain Classifier": Abstain_duration,
                "h Classifier": h_duration,
            }

            with open(os.path.join(args.output_path, 'running_time.json'), 'w') as f:
                json.dump(running_time_dict, f)

            evaluate(args)
            print('Running Time (Integer Programming): {}'.format(IP_duration))
            print('Running Time (Classifier Training): {}'.format(Abstain_duration + h_duration))
            print('Running Time (Total): {}'.format(IP_duration + Abstain_duration + h_duration))

            if os.path.exists(os.path.join(args.output_path, "AbstainClassifier")):
                shutil.rmtree(os.path.join(args.output_path, "AbstainClassifier"))

            if os.path.exists(os.path.join(args.output_path, "hClassifier")):
                shutil.rmtree(os.path.join(args.output_path, "hClassifier"))