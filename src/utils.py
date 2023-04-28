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


    train_data = CustomDataset(np.load(os.path.join(args.data_path, 'X_train.npy')), np.load(os.path.join(args.data_path, 'y_train.npy')))
    test_data = CustomDataset(np.load(os.path.join(args.data_path, 'X_test.npy')), np.load(os.path.join(args.data_path, 'y_test.npy')))

    return train_data, test_data

def process_data(data, OptimalNet, wn, hn):
    X = np.column_stack(
        (data[0].numpy(),
         ((OptimalNet(data[0]) >= 0.5) * 1).detach().cpu().numpy().flatten()
    ))
    return CustomDataset(X, wn), CustomDataset(X, hn)


def load_model(model, path):
    checkpoint = torch.load(
        os.path.join(path, "model_state.pth"),
        map_location="cpu"
    )
    model.load_state_dict(checkpoint)
    model.to("cpu")
    model.eval()
    return model