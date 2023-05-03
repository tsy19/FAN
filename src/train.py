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

    # start_time = time.time()
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_factor, patience=args.lr_patience, verbose=True
    )

    best_loss = float('inf')
    loss_list = []
    for i in range(args.max_epoch):
        if i % 50 == 0 and i != 0:
            # loss_list = np.hstack(loss_list).tolist()
            torch.save(model.state_dict(), os.path.join(path, "model_state.pth"))
            # with open(os.path.join(path, 'loss.txt'), 'a') as file:
            #     for item in loss_list:
            #         file.write(str(item) + '\n')
            # loss_list = []
        val_loss = 0
        model.eval()
        correct, total = 0, 0
        for batch_idx, (features, labels) in enumerate(valloader):
            features, labels = features.to(device=device, dtype=torch.float), labels.to(device=device,
                                                                                        dtype=torch.float)
            model.zero_grad()
            log_probs = model(features).flatten()
            predictions = (log_probs >= 0.5) * 1
            loss = criterion(log_probs, labels)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            val_loss += loss.item()
        val_accuracy = 100 * correct / total
        val_loss /= len(valloader)
        epoch_loss = 0
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
        print("Epoch {}, Training loss {}, Val loss {}, Val accuracy {} %".format(i, epoch_loss, val_loss, val_accuracy))
        with open(os.path.join(path, 'loss.txt'), 'a') as f:
            f.write("Epoch {}, Training loss {}, Val loss {}, Val accuracy {} %\n".format(i, epoch_loss, val_loss,
                                                                                          val_accuracy))
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                if i >= MIN_EPOCHS:
                    print('Early stopping after', i, 'epochs')
                    # loss_list = np.hstack(loss_list).tolist()
                    torch.save(model.state_dict(), os.path.join(path, "model_state.pth"))
                    # with open(os.path.join(path, 'loss.txt'), 'a') as file:
                    #     for item in loss_list:
                    #         file.write(str(item) + '\n')
                    break

    # duration = time.time() - start_time
    # print('Training Time: {}'.format(duration))
    return
