import torch
import torch.nn as nn
from model import TwolayerNet, MLP, AbstainNet
from torch.utils.data import DataLoader
from utils import load_data

import numpy as np
import os
import time
from options import args_parser

def train_abstain(args, train_data):

    if "cuda" in args.device:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device("cpu")

    criterion = nn.BCELoss().to(device)

    trainloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = AbstainNet(args)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0
    )
    
    model_file = os.path.join(args.model_path, "abstain_classifier_" + \
                            args.type_of_abstain+"_" + \
                            args.fairness_notion + "_" + \
                            args.attribute + ".pth")

    if args.load_from_disk == True:
        checkpoint = torch.load(model_file, map_location=device)
#         print(checkpoint.keys())
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.train()

    loss_list = []
    for i in range(args.epoch):
        if i % 10 == 0 and i != 0:
            loss_list = np.hstack(loss_list).tolist()
            
            
            torch.save(
                model.state_dict(), 
                model_file
            )
            
            with open(
                os.path.join(args.model_path, "loss_" + \
                args.type_of_abstain+"_" + \
                args.fairness_notion + "_" + \
                args.attribute + ".txt"), 'a'
            ) as file:
                for item in loss_list:
                    file.write(str(item) + '\n')
            loss_list = []
        losses = 0
        count = 0
        for batch_idx, (features, labels) in enumerate(trainloader):
            features, labels = features.to(device = device, dtype=torch.float), labels.to(device=device, dtype=torch.float)
            model.zero_grad()
            log_probs = model(features).flatten()
            loss = criterion(log_probs, labels)
            losses += loss * labels.shape[0]
            count += labels.shape[0]
            loss.backward()
            optimizer.step()
        epoch_loss = losses / count
        print(i, epoch_loss)
        loss_list.append(epoch_loss.detach().cpu().numpy())



if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    print(args)

    args.model_path = os.path.join("../data/AbstainClassifier")
    args.data_path = "../data/adult"
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    #load w_n and h*_n
    optimal_labels = np.load(os.path.join(args.data_path, "optimal_labels.npy"))
    
    abstain_labels = np.load(
        os.path.join(args.data_path, "abstain_labels_" + \
                args.type_of_abstain+"_" + \
                args.fairness_notion + "_" + \
                args.attribute + ".npy")
    )
    print(optimal_labels.shape, abstain_labels.shape)
    
    train_data, _ = load_data(args, optimal_labels, abstain_labels)
    
    train_abstain(args, train_data)

    duration = time.time() - start_time
    print('Time: {}'.format(duration))