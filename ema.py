import copy
import torch
def build_target(online):
    target=copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad=False
    return target

def update_target(online,target,tau=0.996):
    with torch.no_grad():
        for t,o in zip(target.parameters(),online.parameters()):
            t.mul_(tau).add_(o,alpha=(1-tau))


