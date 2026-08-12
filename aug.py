import torch
from torchvision.transforms import v2

class TwoView:
    def __init__(self):
        super(TwoView, self).__init__()

        self.transform1=v2.Compose([
            v2.RandomResizedCrop(size=(32,32),antialias=True),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(3,3),sigma=(0.1,2.0))],p=1.0),
            v2.RandomApply([v2.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0.1,)],p=0.8),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomGrayscale(p=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32,scale=True),
            v2.RandomSolarize(threshold=0.5,p=0.0),
            v2.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
        ])

        self.transform2=v2.Compose([
            v2.RandomResizedCrop(size=(32,32),antialias=True),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(3,3),sigma=(0.1,2.0))],p=0.1),
            v2.RandomApply([v2.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0.1)],p=0.8),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomGrayscale(p=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32,scale=True),
            v2.RandomSolarize(threshold=0.5,p=0.2),
            v2.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),

        ])

    def __call__(self,img):
        return self.transform1(img),self.transform2(img)



