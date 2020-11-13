import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_features, out_features, l1=128, l2=64):
        super(Net, self).__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fclast = nn.Linear(l2, out_features)

    def forward(self, x):
        #x = x.reshape(-1, x.shape[1] * x.shape[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fclast(x)
        return x