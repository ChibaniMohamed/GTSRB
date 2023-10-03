import torch 
import torch.nn as nn
from torchvision.datasets import GTSRB
from torchvision.transforms import Compose,ToTensor,Resize,Normalize,RandomRotation,ColorJitter,RandomHorizontalFlip,Pad,AutoAugment
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transforms = Compose([
    AutoAugment(),
    Resize([50,50]),
    ToTensor(),
    
])

train_dataset = GTSRB(root='./gtsrb_dataset/',split='train',transform=transforms)

train_dataloader = DataLoader(dataset=train_dataset,shuffle=True)
print(len(train_dataloader))
for step,(input,label) in enumerate(train_dataloader):
    conv1 = nn.Conv2d(3,16,3)(input).detach()
    conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)(conv1).detach()
    maxpool1 = nn.MaxPool2d(2)(conv2).detach()
    batchnorm1 = nn.BatchNorm2d(32)(maxpool1).detach()

    conv3 =  nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)(batchnorm1).detach()
    conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)(conv3).detach()
    maxpool2 = nn.MaxPool2d(2)(conv4).detach()
    batchnorm2 = nn.BatchNorm2d(128)(maxpool2).detach()

    
    
    leakyrelu = nn.LeakyReLU()(conv3)
    relu = nn.ReLU()(conv3)
    dropout = nn.Dropout2d()(conv3)
    print(leakyrelu.shape,relu.shape)
    plt.figure(1)
    plt.imshow(input[0].permute(1,2,0))
    fig,axes = plt.subplots(ncols=10,nrows=3,figsize=(10,10))
    for i,(feature1,feature2,maxpl1) in enumerate(zip(maxpool1[0],batchnorm1[0],batchnorm2[0])):
            if(i == 10):
                  break
            '''
            if i == 0:
                feature = matrix[0].permute(1,2,0)
            '''
        
            axes[0,i].imshow(feature1,cmap='inferno')
            axes[1,i].imshow(feature2,cmap='inferno')
            axes[2,i].imshow(maxpl1,cmap='inferno')



    plt.show()