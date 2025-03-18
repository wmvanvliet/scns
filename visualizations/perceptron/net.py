import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28 + 1, 512, bias=False)
        self.fc2 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = torch.cat((torch.flatten(x, 1), torch.ones((len(x), 1), device=x.device)), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
