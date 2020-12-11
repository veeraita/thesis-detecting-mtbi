import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_features, out_features, l1=128, l2=64, l3=None):
        super(Net, self).__init__()
        self.in_features = in_features
        self.bn = nn.BatchNorm1d(l1)
        self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features, l1)
        self.fc2 = nn.Linear(l1, l2)
        if l3 is not None:
            self.fc3 = nn.Linear(l2, l3)
            self.fclast = nn.Linear(l3, out_features)
        else:
            self.fc3 = None
            self.fclast = nn.Linear(l2, out_features)

    def forward(self, x):
        x = self.drop(x)
        x = F.relu(self.bn(self.fc1(x)))
        x = F.relu(self.fc2(x))
        if self.fc3 is not None:
            x = F.relu(self.fc3(x))
        x = self.fclast(x)
        return x