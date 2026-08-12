from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        m=resnet18(weights=None)
        m.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
        m.maxpool=nn.Identity()
        m.fc=nn.Identity()
        self.net=m

    def forward(self,x):
        return self.net(x)


class Projector(nn.Module):
    def __init__(self):
        super(Projector,self).__init__()

        self.fc1=nn.Linear(512,4096)
        self.bn1=nn.BatchNorm1d(4096)
        self.fc2=nn.Linear(4096,256)

    def forward(self,x):
        h=F.relu(self.bn1(self.fc1(x)))
        h=self.fc2(h)
        return h

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor,self).__init__()

        self.fc1=nn.Linear(256,4096)
        self.bn1=nn.BatchNorm1d(4096)
        self.fc2=nn.Linear(4096,256)

    def forward(self,x):
        h=F.relu(self.bn1(self.fc1(x)))
        h=self.fc2(h)
        return h


class Base(nn.Module):
    def __init__(self):
        super(Base,self).__init__()

        self.encoder=Encoder()
        self.projector=Projector()

    def forward(self,x):
        return self.projector(self.encoder(x))