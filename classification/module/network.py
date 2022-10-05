import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models


###############################
# ## for USPS <-> MNIST
# ##############################

class TinyG(nn.Module):
    def __init__(self, prob=0.1):
        super(TinyG, self).__init__()
        self.p = prob
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)
        self._initialize_weights()

    def forward(self, x, dropout=True):
        x = torch.mean(x,1).view(x.size()[0],1,x.size()[2],x.size()[3])
        # b x 1 x 28 x 28
        x = self.bn1(self.conv1(x))
        # b x 32 x 24 x 24
        x = F.max_pool2d(F.relu(x), 2)
        # b x 32 x 12 x 12
        x = F.dropout2d(x, p=self.p, training=dropout)
        x = self.bn2(self.conv2(x))
        # b x 48 x 8 x 8
        x = F.max_pool2d(F.relu(x), 2)
        # b x 48 x 4 x 4
        x = x.view(x.size(0), 48*4*4)
        # b x 48*4*4
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class TinyH(nn.Module):
    def __init__(self, prob=0.4):
        super(TinyH, self).__init__()
        self.p = prob
        self.fc1 = nn.Linear(48*4*4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self._initialize_weights()

    def forward(self, x, dropout=True):
        x = F.dropout(x, p=self.p, training=dropout)
        # b x 48*4*4
        x = F.relu(self.bn1_fc(self.fc1(x)))
        # b x 100
        x = F.dropout(x, p=self.p, training=dropout)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        # b x 100
        x = F.dropout(x, p=self.p, training=dropout)
        x = self.fc3(x)
        # b x 10
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


###############################
# ## for SVHN -> MNIST
# ##############################

class MidG(nn.Module):
    def __init__(self, prob=0.1):
        super(MidG, self).__init__()
        self.p = prob
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self._initialize_weights()

    def forward(self, x, dropout=True):
        # b x 32 x 32 x 3
        x = self.bn1(self.conv1(x))
        # b x 32 x 32 x 64
        x = F.max_pool2d(F.relu(x), 2)
        # b x 16 x 16 x 64
        x = F.dropout2d(x, p=self.p, training=dropout)
        x = self.bn2(self.conv2(x))
        # b x 16 x 16 x 64
        x = F.max_pool2d(F.relu(x), 2)
        # b x 8 x 8 x 64
        x = F.dropout2d(x, p=self.p, training=dropout)
        x = F.relu(self.bn3(self.conv3(x)))
        # b x 8 x 8 x 128
        x = F.dropout2d(x, p=self.p, training=dropout)
        x = x.view(x.size(0), 8192)
        # b x 8192
        x = F.relu(self.bn1_fc(self.fc1(x)))
        # b x 3072
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

class MidH(nn.Module):
    def __init__(self, prob=0.4):
        super(MidH, self).__init__()
        self.p = prob
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self._initialize_weights()

    def forward(self, x, dropout=True):
        x = F.dropout(x, p=self.p, training=dropout)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        # b x 2048
        x = F.dropout(x, p=self.p, training=dropout)
        x = self.fc3(x)
        # b x 10
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


###############################
# ## for Synsig -> GTSRB
# ##############################

class BigG(nn.Module):
    def __init__(self, prob=0.1):
        super(BigG, self).__init__()
        self.p = prob
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(144)
        self.conv3 = nn.Conv2d(144, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        self._initialize_weights()

    def forward(self, x, dropout=True):
        # b x 40 x 40 x 3
        x = self.bn1(self.conv1(x))
        # b x 40 x 40 x 96
        x = F.max_pool2d(F.relu(x), 2)
        # b x 20 x 20 x 96
        x = F.dropout2d(x, p=self.p, training=dropout)
        x = self.bn2(self.conv2(x))
        # b x 20 x 20 x 144
        x = F.max_pool2d(F.relu(x), 2)
        # b x 10 x 10 x 144
        x = F.dropout2d(x, p=self.p, training=dropout)
        x = self.bn3(self.conv3(x))
        # b x 10 x 10 x 256
        x = F.max_pool2d(F.relu(x), 2)
        # b x 5 x 5 x 256
        x = F.dropout2d(x, p=self.p, training=dropout)
        x = x.view(x.size(0), 6400)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BigH(nn.Module):
    def __init__(self, prob=0.4):
        super(BigH, self).__init__()
        self.p = prob
        self.fc2 = nn.Linear(6400, 512)
        self.bn2_fc = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 43)
        self.bn_fc3 = nn.BatchNorm1d(43)
        self._initialize_weights()

    def forward(self, x, dropout=True):
        # b x 6400
        x = F.relu(self.bn2_fc(self.fc2(x)))
        # b x 512
        x = F.dropout(x, p=self.p, training=dropout)
        x = self.fc3(x)
        # b x 43
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


##################################
### for Office-31, Office-Home
##################################

class R50G(nn.Module):
    def __init__(self, pretrained=False):
        super(R50G, self).__init__()
        model_resnet = models.resnet50(pretrained=False)
        if pretrained:
            pretrained_weights = 'model/resnet50-0676ba61.pth'
            model_resnet.load_state_dict(torch.load(pretrained_weights, map_location=lambda storage, loc:storage))
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        #self.in_features = model_resnet.fc.in_features

    def forward(self, x, dropout=False, p1=0.0, p2=0.1, p3=0.1, p4=0.5):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(F.dropout(x, p=p1, training=dropout))
        x = self.layer2(F.dropout(x, p=p2, training=dropout))
        x = self.layer3(F.dropout(x, p=p3, training=dropout))
        x = self.layer4(F.dropout(x, p=p4, training=dropout))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class R50H(nn.Module):
    def __init__(self, prob=0.4, nb_class=31):
        super(R50H, self).__init__()
        self.p = prob
        self.n_class = nb_class
        self.fc1 = nn.Linear(2048, 1000, bias=True)
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000, bias=True)
        self.bn2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, self.n_class, bias=True)
        self._initialize_weights()

    def forward(self, x, reverse=False, alpha=0.1, dropout=False):
        if reverse:
            x = rev_grad(x, alpha)
        x = F.dropout(x, p=self.p, training=dropout)
        x = self.bn1(self.fc1(x))
        x = F.relu(F.dropout(x, p=self.p, training=dropout))
        x = self.bn2(self.fc2(x))
        x = F.relu(F.dropout(x, p=self.p, training=dropout))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


###############################
#### for VidDA-17
################################

class R101G(nn.Module):
    def __init__(self, pretrained=False):
        super(R101G, self).__init__()
        model_resnet = models.resnet101(pretrained=False)
        if pretrained:
            pretrained_weights = 'model/resnet101-63fe2227.pth'
            model_resnet.load_state_dict(torch.load(pretrained_weights, map_location=lambda storage, loc:storage))
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        #self.in_features = model_resnet.fc.in_features

    def forward(self, x, dropout=False, p1=0.0, p2=0.1, p3=0.1, p4=0.5):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(F.dropout(x, p=p1, training=dropout))
        x = self.layer2(F.dropout(x, p=p2, training=dropout))
        x = self.layer3(F.dropout(x, p=p3, training=dropout))
        x = self.layer4(F.dropout(x, p=p4, training=dropout))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class R101H(nn.Module):
    def __init__(self, n_class=12, prob=0.5):
        super(R101H, self).__init__()
        self.p = prob
        self.fc1 = nn.Linear(2048, 1000, bias=True)
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000, bias=True)
        self.bn2 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, n_class, bias=True)
        self._initialize_weights()

    def forward(self, x, reverse=False, alpha=0.1, dropout=False):
        if reverse:
            x = rev_grad(x, alpha)
        x = self.bn1(self.fc1(x))
        x = F.relu(F.dropout(x, p=self.p, training=dropout))
        x = self.bn2(self.fc2(x))
        x = F.relu(F.dropout(x, p=self.p, training=dropout))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
