from torch.utils.data import Dataset
import torch
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def load_data(args):
    x = np.load(os.path.join(args.data_path, 'X_train.npy'))
    y = np.load(os.path.join(args.data_path, 'y_train.npy'))
    n = x.shape[0]
    truncated = int(0.8 * n)
    x_train = x[:truncated]
    y_train = y[:truncated]
    x_val = x[truncated:]
    y_val = y[truncated:]
    train_data = CustomDataset(x_train, y_train)
    val_data = CustomDataset(x_val, y_val)
    test_data = CustomDataset(np.load(os.path.join(args.data_path, 'X_test.npy')), np.load(os.path.join(args.data_path, 'y_test.npy')))

    return train_data, val_data, test_data

def process_data(data, OptimalNet, wn, hn):
    X = np.column_stack(
        (data[0].numpy(),
         ((OptimalNet(data[0]) >= 0.5) * 1).detach().cpu().numpy().flatten()
    ))
    n = X.shape[0]
    truncated = int(0.8 * n)
    X_train = X[:truncated]
    wn_train = wn[:truncated]
    hn_train = hn[:truncated]
    X_val = X[truncated:]
    wn_val = wn[truncated:]
    hn_val = hn[truncated:]
    return CustomDataset(X_train, wn_train), CustomDataset(X_train, hn_train), \
           CustomDataset(X_val, wn_val), CustomDataset(X_val, hn_val)


def load_model(model, path):
    checkpoint = torch.load(
        os.path.join(path, "model_state.pth"),
        map_location="cpu"
    )
    model.load_state_dict(checkpoint)
    model.to("cpu")
    model.eval()
    return model