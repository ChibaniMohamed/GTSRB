import torch
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB
from torchvision.transforms import ToTensor,Compose

transforms = Compose([ToTensor()])

train_dataset = GTSRB(root='./gtsrb_dataset/',split='train',transform=transforms)
test_dataset = GTSRB(root='./gtsrb_dataset/',split='test',transform=transforms)

train_loader = DataLoader(train_dataset)
test_loader = DataLoader(test_dataset)



print('training data :',len(train_loader))




print('testing data :',len(test_loader))