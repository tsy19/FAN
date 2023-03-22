import torch
import torch.nn as nn
from model import TwolayerNet, MLP
from torch.utils.data import DataLoader
from utils import load_data

import numpy as np
import os
import time
from options import args_parser

def train_optimal(args, train_data):

    device = args.device
    criterion = nn.CrossEntropyLoss().to(device)

    trainloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = MLP(args)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    if args.load_from_disk == True:
        model.load_state_dict(torch.load(os.path.join(args.model_path, "model_state.pth")))

    loss_list = []
    for i in range(args.epoch):
        if i % 50 == 0 and i != 0:
            loss_list = np.hstack(loss_list).tolist()
            torch.save(model.state_dict(), os.path.join(args.model_path, "model_state.pth"))
            with open(os.path.join(args.model_path, 'loss.txt'), 'a') as file:
                for item in loss_list:
                    file.write(str(item) + '\n')
            loss_list = []
        losses = 0
        count = 0
        for batch_idx, (features, labels) in enumerate(trainloader):
            features, labels = features.to(device = device, dtype=torch.float), labels.to(device=device, dtype=torch.long)
            model.zero_grad()
            log_probs = model(features)
            loss = criterion(log_probs, labels)
            losses += loss * labels.shape[0]
            count += labels.shape[0]
            loss.backward()
            optimizer.step()
        epoch_loss = losses / count
        print(epoch_loss)
        loss_list.append(epoch_loss.detach().cpu().numpy())



if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    print(args)

    args.model_path = "../data/OptimalClassifier"
    args.data_path = "../data/adult"

    train_data, test_data = load_data(args)

    train_optimal(args, train_data)

    duration = time.time() - start_time
    print('Time: {}'.format(duration))