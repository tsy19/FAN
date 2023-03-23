import torch
import torch.nn as nn
from model import TwolayerNet, MLP
from torch.utils.data import DataLoader
from utils import load_data

import numpy as np
import os
import time
from options import args_parser

def load_optimal(args):
    model = MLP(args)

    checkpoint = torch.load(os.path.join(args.model_path, "model_state.pth"), map_location="cpu")
    model.load_state_dict(checkpoint)

    model.to("cpu")
    model.eval()

    return model

def predict(model, data):
    log_probs = model(data.X)
    return (log_probs >= 0.5) * 1, log_probs

def train_abstain(args, train_data):

    #load optimal classifier
    optimal_classifier = load_optimal(args)
    pred_labels, pred_probs = predict(optimal_classifier, train_data)
    a = 1





if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    print(args)

    args.model_path = "../data/OptimalClassifier"
    args.data_path = "../data/adult"

    train_data, test_data = load_data(args)

    train_abstain(args, train_data)

    duration = time.time() - start_time
    print('Time: {}'.format(duration))