import torch
import os
import json
from aug import TwoView
from ema import build_target,update_target
from loss import byol_loss
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim import Adam
from networks import Base,Predictor
from tqdm import tqdm


os.makedirs("checkpoint",exist_ok=True)
if __name__ == "__main__":
    dataset=CIFAR10(root='./data',train=True,download=True,transform=TwoView())
    loader=DataLoader(dataset,batch_size=256,shuffle=True,num_workers=2,drop_last=True)
    device="mps" if torch.backends.mps.is_available() else "cpu"

    online_base=Base().to(device)
    predictor=Predictor().to(device)
    target_base=build_target(online_base).to(device)

    optimizer=Adam(
        list(online_base.parameters())+list(predictor.parameters()),
        lr=3e-4,
    )

    online_base.train()
    predictor.train()
    loss_history=[]
    num_epochs=100
    for epoch in range(num_epochs):
        total_loss=0
        for (v1,v2),_ in tqdm(loader,desc=f"Epoch {epoch:3d}",leave=False):
            v1=v1.to(device)
            v2=v2.to(device)
            p1=predictor(online_base(v1))
            p2=predictor(online_base(v2))
            with torch.no_grad():
                z1=target_base(v1)
                z2=target_base(v2)
            loss=byol_loss(p1,z2)+byol_loss(p2,z1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_target(online_base,target_base)
            total_loss+=loss.item()

        avg_loss=total_loss/len(loader)
        loss_history.append(avg_loss)
        print(f"[epoch {epoch+1:3d}/{num_epochs}]avg_loss:{avg_loss:.4f}")
        save_every=10
        if (epoch+1)%save_every==0:
            torch.save(online_base.encoder.state_dict(),f"./checkpoint/byol_encoder_ep{epoch+1}.pth")
            tqdm.write(f"Saved checkpoint at epoch {epoch+1}")

    torch.save(online_base.encoder.state_dict(),"./checkpoint/byol_encoder.pth")
    print("Saved Encoded Checkpoint")
    with open(f"./checkpoint/loss_history.json",'w') as f:
        json.dump(loss_history,f)
