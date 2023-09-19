import torch
import torch.nn as nn
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam
from torchvision.transforms import ToTensor,Resize,Compose
import matplotlib.pyplot as plt
import tqdm
BATCH_SIZE = 4
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
transforms = Compose([
Resize([112,112]),
ToTensor()
])

train_dataset = GTSRB(root='./gtsrb_dataset/',split="train",transform=transforms)
train_loader = DataLoader(dataset=train_dataset,shuffle=True)


class GTSRB_NETWORK(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(GTSRB_NETWORK,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
      

        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout(0.2)
        self.dropout2D = nn.Dropout2d(0.2)

        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()


        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv3 =  nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(2)
        self.batchnorm2 = nn.BatchNorm2d(128)

        self.conv5 =  nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(2)
        self.batchnorm3 = nn.BatchNorm2d(512)

        self.l1 = nn.Linear(512*10*10,245)
        
        self.l2 = nn.Linear(245,128)
        self.l3 = nn.Linear(128,64)
        self.l4 = nn.Linear(64,output_dim)
        
    def forward(self,input):
        
        conv = self.conv1(input)
        conv = self.conv2(conv)
        maxpool = self.maxpool1(conv)
        batchnorm = self.batchnorm1(maxpool)

        conv = self.conv3(batchnorm)
        conv = self.conv4(conv)
        maxpool = self.maxpool2(conv)
        batchnorm = self.batchnorm2(maxpool)

        conv = self.conv5(batchnorm)
        conv = self.conv6(conv)
        maxpool = self.maxpool3(conv)
        batchnorm = self.batchnorm3(maxpool)

        flatten = self.flatten(batchnorm)
        
        dense_l1 = self.l1(flatten)
        dense_l2 = self.l2(dense_l1)
        dense_l3 = self.l3(dense_l2)
        output = self.l4(dense_l3) 
       
        return output
    
EPOCHS = 8
LEARNING_RATE = 0.001
INPUT_DIM = 3*112*112
OUTPUT_DIM = 43
STEPS = len(train_loader)
model = GTSRB_NETWORK(INPUT_DIM,OUTPUT_DIM).to(device)
optimizor = Adam(params=model.parameters(),lr=LEARNING_RATE)
loss = nn.CrossEntropyLoss()
try:
    for epoch in range(EPOCHS):
        true_pred = 0
        with tqdm.trange(STEPS) as STEPS_:
            for step,(input,label) in enumerate(train_loader,start=1):
                input,label = input.to(device),label.to(device)
                prediction = model.forward(input)
                if prediction.argmax(1)[0] == label[0] :
                    true_pred += 1
                l = loss(prediction,label)
                l.backward()
                optimizor.step()
                optimizor.zero_grad()
                STEPS_.colour = 'green'
                STEPS_.desc = f'Epoch [{epoch}/{EPOCHS}], Step [{step}/{STEPS}], Loss [{"{:.3f}".format(l)}], Accuracy [{"{:.3f}".format(true_pred/step)}]'
                STEPS_.update(1)
    torch.jit.script(model).save('./models/gtsrb_model.pt')
except KeyboardInterrupt:
    torch.jit.script(model).save('./models/gtsrb_model.pt')

model = model.eval()
for i,(input,label) in enumerate(train_loader):
    
    normalized_input = input[0].permute(1,2,0)
    prediction = model.forward(input.to(device)).argmax(1)[0]
    plt.title(f'predicted :{prediction} | ground truth : {label[0]}')
    plt.imshow(normalized_input)
    plt.show()
    
