import torch.nn as nn
import torch

class TwolayerNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        torch.manual_seed(args.seed)
        self.cls = torch.nn.Sequential(
          nn.Linear(args.input_dim, args.hidden_dim),
          nn.ReLU(),
          nn.Dropout(),
          nn.Linear(args.hidden_dim, 2),
          nn.Softmax(dim=1)
        )
        for layer in self.cls:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, 0, 0.01)
                nn.init.constant_(layer.bias.data, 0)

    def forward(self, input):
        output = self.cls(input)
        return output


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()

        input_dim = args.input_dim
        hidden_dim = args.hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
class AbstainNet(nn.Module):
    def __init__(self, args):
        super(AbstainNet, self).__init__()

        input_dim = args.input_dim + 1
        hidden_dim = args.hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x