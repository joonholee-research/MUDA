import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkF(nn.Module):
    def __init__(self):
        super(NetworkF, self).__init__()
        self.f_fc1 = nn.Linear(2, 15)
        self.f_bn1 = nn.BatchNorm1d(15)
        self.f_fc2 = nn.Linear(15, 15)
        self.f_bn2 = nn.BatchNorm1d(15)
        self.f_fc3 = nn.Linear(15, 15)
        self.f_bn3 = nn.BatchNorm1d(15)
        self._initialize_weights()
        
    def forward(self, x):
        x = F.relu(self.f_bn1(self.f_fc1(x)))
        x = F.relu(self.f_bn2(self.f_fc2(x)))
        x = F.relu(self.f_bn3(self.f_fc3(x)))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

                
class NetworkC(nn.Module):
    def __init__(self):
        super(NetworkC, self).__init__()
        self.c_fc1 = nn.Linear(15, 15)
        self.c_fc2 = nn.Linear(15, 2)
        self._initialize_weights()

    def forward(self, x, dropout=True):
        x = self.c_fc1(x)
        x = F.relu(F.dropout(x, p=0.5, training=dropout))
        x = self.c_fc2(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
  