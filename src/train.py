import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import os

def train_model(train_data, val_data, model, args, path=""):
    MIN_EPOCHS = args.min_epoch
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
    valloader = DataLoader(
        val_data,
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


    best_loss = float('inf')
    loss_list = []
    for i in range(args.max_epoch):
        if i % 50 == 0 and i != 0:
            loss_list = np.hstack(loss_list).tolist()
            torch.save(model.state_dict(), os.path.join(path, "model_state.pth"))
            with open(os.path.join(path, 'loss.txt'), 'a') as file:
                for item in loss_list:
                    file.write(str(item) + '\n')
            loss_list = []
        epoch_loss = 0
        count = 0
        model.train()
        for batch_idx, (features, labels) in enumerate(trainloader):
            features, labels = features.to(device=device, dtype=torch.float), labels.to(device=device, dtype=torch.float)
            model.zero_grad()
            log_probs = model(features).flatten()
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(trainloader)
        print(i, epoch_loss)
        loss_list.append(epoch_loss)
        val_loss = 0
        model.eval()
        for batch_idx, (features, labels) in enumerate(valloader):
            features, labels = features.to(device=device, dtype=torch.float), labels.to(device=device,
                                                                                        dtype=torch.float)
            model.zero_grad()
            log_probs = model(features).flatten()
            loss = criterion(log_probs, labels)
            val_loss += loss.item()
        val_loss /= len(valloader)

        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                if i >= MIN_EPOCHS:
                    print('Early stopping after', i, 'epochs')
                    loss_list = np.hstack(loss_list).tolist()
                    torch.save(model.state_dict(), os.path.join(path, "model_state.pth"))
                    with open(os.path.join(path, 'loss.txt'), 'a') as file:
                        for item in loss_list:
                            file.write(str(item) + '\n')
                    break

    duration = time.time() - start_time
    print('Training Time: {}'.format(duration))
    return
