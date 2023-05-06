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

def process_data(data, args, OptimalNet, train=True, wn=None, hn=None):
    X = np.column_stack(
        (data.X.numpy(),
         (OptimalNet(data.X).detach().cpu().numpy().flatten())
         ))
    n = X.shape[0]
    if train == True:
        truncated = int(n * args.sample_ratio)
        X_train = X[:truncated]
        wn_train = wn[:truncated]
        hn_train = hn[:truncated]
        X_val = X[truncated:]
        wn_val = wn[truncated:]
        hn_val = hn[truncated:]
        return CustomDataset(X_train, wn_train), CustomDataset(X_train, hn_train), \
               CustomDataset(X_val, wn_val), CustomDataset(X_val, hn_val)
    else:
        return CustomDataset(X, data.y.numpy())



def load_model(model, path):
    checkpoint = torch.load(
        os.path.join(path, "model_state.pth"),
        map_location="cpu"
    )
    model.load_state_dict(checkpoint)
    model.to("cpu")
    model.eval()
    return model

def sample_data(data, sample_length):
    sampled_data = data[:sample_length]
    X = sampled_data[0].numpy()
    y = sampled_data[1].numpy().astype(int)
    return CustomDataset(X, y)

def compute_stats(data, args, wn, hn, star_n=None):
    X = data.X.numpy()
    y = data.y.numpy()

    un = wn * hn
    group_indices = args.group_indices if args.dataset == "adult" else [1, 0]
    g_v = []
    stats = {
        "num_of_group": len(group_indices),
        "num_of_data": [],
        "num_of_positive_data": [],
        "num_of_negative_data": [],
        "raw_stats": [],
        "accept_rate": [],
        "accept_rate_no_abstain": [],
        "true_positive_rate": [],
        "true_positive_rate_no_abstain": [],
        "false_negative_rate": [],
        "false_negative_rate_no_abstain": [],
        "true_negative_rate": [],
        "true_negative_rate_no_abstain": [],
        "false_positive_rate": [],
        "false_positive_rate_no_abstain": [],
        "accuracy": [],
        "error": [],
        "abstain_rate": [],
        "abstain_rate_positive": [],
        "abstain_rate_negative": [],
    }
    if star_n is not None:
        stats["flip_rate"] = []
        stats["flip_rate_positive"] = []
        stats["flip_rate_negative"] = []
    for idx in range( len(group_indices) + 1):
        if idx == 0:
            temp = np.ones(y.shape[0])
        else:
            if args.dataset == 'adult':
                item = group_indices[idx - 1]
                temp = (X[:, item] == 1) * 1
            else:
                item = group_indices[idx - 1]
                temp = (X[:, args.group_indices] == item) * 1
        g_v.append(temp)
        num_of_data = np.sum(temp)
        num_of_positive_data = np.sum(temp * y)
        num_of_negative_data = np.sum(temp * (1 - y))
        g1 = np.sum(wn * (1 - un) * y * temp)
        g2 = np.sum(wn * un * y * temp)
        g3 = np.sum((1 - wn) * y * temp)
        g4 = np.sum(wn * un * (1 - y) * temp)
        g5 = np.sum(wn * (1 - un) * (1 - y) * temp)
        g6 = np.sum((1 - wn) * (1 - y) * temp)
        if g1 + g2 + g3 != num_of_positive_data:
            print("false1")
        if g4 + g5 + g6 != num_of_negative_data:
            print("false2")
        if num_of_negative_data + num_of_positive_data != num_of_data:
            print("false3")

        stats['num_of_data'].append(num_of_data)
        stats['num_of_positive_data'].append(num_of_positive_data)
        stats['num_of_negative_data'].append(num_of_negative_data)
        stats["raw_stats"].append([g1, g2, g3, g4, g5, g6])
        stats["accept_rate"].append( (g2 + g4) / num_of_data )
        stats['accept_rate_no_abstain'].append( (g2 + g4) / (g1 + g2 + g4 + g5) )
        stats['true_positive_rate'].append( g2 / num_of_positive_data )
        stats['true_positive_rate_no_abstain'].append( g2 / (g1 + g2))
        stats['false_negative_rate'].append( g1 / num_of_positive_data )
        stats['false_negative_rate_no_abstain'].append(g1 / (g1 + g2))
        stats['true_negative_rate'].append(g5 / num_of_negative_data)
        stats['true_negative_rate_no_abstain'].append(g5 / (g4 + g5))
        stats['false_positive_rate'].append(g4 / num_of_negative_data)
        stats['false_positive_rate_no_abstain'].append(g4 / (g4 + g5))
        stats['accuracy'].append( (g2 + g5) / (g1 + g2 + g4 + g5) )
        stats['error'].append((g1 + g4) / (g1 + g2 + g4 + g5))
        stats['abstain_rate'].append( (g3 + g6) / num_of_data)
        stats['abstain_rate_positive'].append(g3 / num_of_positive_data)
        stats['abstain_rate_negative'].append(g6 / num_of_negative_data)

        if star_n is not None:
            stats['flip_rate'].append( np.sum((hn != star_n) * temp) / num_of_data)
            stats['flip_rate_positive'].append( np.sum( (hn != star_n) * y * temp ) / num_of_positive_data)
            stats['flip_rate_negative'].append(np.sum((hn != y) * (1 - y) * temp) / num_of_negative_data)

    return stats

def get_stats(data, args, AbstainNet, HNet, OptimalNet=None, wn=None, hn=None):
    y = data.y.numpy()
    if OptimalNet != None:
        star_n = ((OptimalNet(data.X) >= 0.5) * 1).numpy().flatten()
        optimal_stats = compute_stats(data, args, np.ones(y.shape[0]), star_n, star_n=star_n)
    else:
        star_n = None
    data = process_data(data, args, OptimalNet, False)
    wn_predict = ((AbstainNet(data.X) >= 0.5) * 1).numpy().flatten()
    hn_predict = ((HNet(data.X) >= 0.5) * 1).numpy().flatten()
    hn_predict[wn_predict == 0] = 0
    stats = compute_stats(data, args, wn_predict, hn_predict, star_n=star_n)

    if wn is not None:
        sampled_length = wn.shape[0]
        if sampled_length != y.shape[0]:
            data = sample_data(data, sampled_length)
        hn[wn == 0] = 0
        stats_raw = compute_stats(data, args, wn, hn, star_n=star_n)
        return optimal_stats, stats, stats_raw
    else:
        return optimal_stats, stats
