from torch.utils.data import Dataset
import torch
import numpy as np
import os

class AdultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def load_data(args, optimal_labels=None, abstain_labels=None):
    # Load training and testing data as DataLoader
    if optimal_labels is None:
        train_data = AdultDataset(np.load(os.path.join(args.data_path, 'X_train.npy')), np.load(os.path.join(args.data_path, 'y_train.npy')))
        test_data = AdultDataset(np.load(os.path.join(args.data_path, 'X_test.npy')), np.load(os.path.join(args.data_path, 'y_test.npy')))

        return train_data, test_data
    else:
        X_train = np.load(os.path.join(args.data_path, 'X_train.npy'))
        print(X_train.shape)
        X_train = np.column_stack((X_train, optimal_labels))
        print(X_train.shape)
        
        train_data = AdultDataset(X_train, abstain_labels)
        return train_data, None
        