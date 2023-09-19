import torch 
import torch.nn as nn
from torchvision.datasets import GTSRB
from torchvision.transforms import Compose,ToTensor,Resize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transforms = Compose([
    Resize([112,112]),
    ToTensor()
])

train_dataset = GTSRB(root='./gtsrb_dataset/',split='train',transform=transforms)

train_dataloader = DataLoader(dataset=train_dataset,shuffle=True)

for step,(input,label) in enumerate(train_dataloader):
    conv1 = nn.Conv2d(3,48,3)(input).detach()
    conv2 = nn.Conv2d(in_channels=48,out_channels=128,kernel_size=3)(conv1).detach()
    maxpool1 = nn.MaxPool2d(2)(conv2).detach()

    conv3 =  nn.Conv2d(in_channels=128,out_channels=245,kernel_size=3)(maxpool1).detach()
    conv4 = nn.Conv2d(in_channels=245,out_channels=328,kernel_size=3)(conv3).detach()
    maxpool2 = nn.MaxPool2d(2)(conv4).detach()

    conv5 = nn.Conv2d(in_channels=328,out_channels=400,kernel_size=3)(maxpool2).detach()
    conv6 = nn.Conv2d(in_channels=400,out_channels=480,kernel_size=3)(conv5).detach()
    maxpool3 = nn.MaxPool2d(1)(conv6).detach()
    
    leakyrelu = nn.LeakyReLU()(conv3)
    relu = nn.ReLU()(conv3)
    dropout = nn.Dropout2d()(conv3)
    print(leakyrelu.shape,relu.shape)
    plt.figure(1)
    plt.imshow(input[0].permute(1,2,0))
    fig,axes = plt.subplots(ncols=10,nrows=3,figsize=(10,10))
    for i,(feature1,feature2,maxpl1) in enumerate(zip(maxpool3[0],leakyrelu[0],dropout[0])):
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