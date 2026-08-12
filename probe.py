import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from networks import Encoder


def train_probe(encoder,classifier,train_loader,optimizer,epochs,device):
    classifier.train()
    for epoch in range(epochs):
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            with torch.no_grad():
                features=encoder(images)
            logits=classifier(features)
            loss=F.cross_entropy(logits,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate(encoder,classifier,test_loader,device):
    encoder.eval()
    classifier.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            logits=classifier(encoder(images))
            pred=logits.argmax(dim=1)
            correct+=(pred==labels).sum().item()
            total+=labels.size(0)

        return 100*correct/total


if __name__=='__main__':
    device="mps" if torch.backends.mps.is_available() else "cpu"
    enc=Encoder().to(device)
    enc.load_state_dict(torch.load("./checkpoint/byol_encoder.pth",map_location=device))
    for p in enc.parameters():
        p.requires_grad=False

    enc.eval()
    classifier=nn.Linear(512,10).to(device)
    optimizer=Adam(classifier.parameters(),lr=3e-4)

    tfm=v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale=True),
        v2.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ])

    train_set=CIFAR10(root="./data",train=True,transform=tfm,download=True)
    test_set=CIFAR10(root="./data",train=False,transform=tfm,download=True)
    train_loader=DataLoader(train_set,batch_size=256,shuffle=True,num_workers=2,drop_last=True)
    test_loader=DataLoader(test_set,batch_size=256,shuffle=False,num_workers=2)

    train_probe(enc,classifier,train_loader,optimizer,epochs=15,device=device)
    accuracy=evaluate(enc,classifier,test_loader,device)
    print(f"probe_accuracy:{accuracy:.2f}")
