import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models import MLP
import time
import numpy as np
import os

def train_model(train_data, model, args, path=""):
    if not os.path.exists(path):
        os.makedirs(path)

    start_time = time.time()
    print("Model save to ", path)
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

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0
    )
    #
    # if args.load_from_disk == True:
    #     checkpoint = torch.load(os.path.join(args.model_path, "model_state.pth"), map_location=device)
    #     print(checkpoint.keys())
    #     model.load_state_dict(checkpoint)

    model.train()

    loss_list = []
    for i in range(args.max_epoch):
        if i % 50 == 0 and i != 0:
            loss_list = np.hstack(loss_list).tolist()
            torch.save(model.state_dict(), os.path.join(path, "model_state.pth"))
            with open(os.path.join(path, 'loss.txt'), 'a') as file:
                for item in loss_list:
                    file.write(str(item) + '\n')
            if loss_list[-2] - loss_list[-1] <= args.stopping_ratio * loss_list[0]:
                duration = time.time() - start_time
                print('Training Time: {}'.format(duration))
                return
            loss_list = []
        losses = 0
        count = 0
        for batch_idx, (features, labels) in enumerate(trainloader):
            features, labels = features.to(device=device, dtype=torch.float), labels.to(device=device, dtype=torch.float)
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
    duration = time.time() - start_time
    print('Training Time: {}'.format(duration))
    return
