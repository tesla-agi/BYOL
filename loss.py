import torch.nn.functional as F

def byol_loss(p,z):
    p=F.normalize(p,dim=-1)
    z=F.normalize(z,dim=-1)
    return (2-2*(p*z).sum(dim=-1)).mean()

