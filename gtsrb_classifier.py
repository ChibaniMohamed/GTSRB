import torch
import torch.nn as nn
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
from torch.optim import SGD
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
        
        self.maxpool = nn.MaxPool2d(2)
        self.maxpoolLast = nn.MaxPool2d(1)

        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout(0.2)
        self.dropout2D = nn.Dropout2d(0.2)

        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()


        self.conv1 = nn.Conv2d(3,48,3)
        self.conv2 = nn.Conv2d(in_channels=48,out_channels=128,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=245,kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=245,out_channels=328,kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=328,out_channels=400,kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=400,out_channels=480,kernel_size=3)
        self.conv7 = nn.Conv2d(in_channels=480,out_channels=528,kernel_size=3)
        self.conv8 = nn.Conv2d(in_channels=528,out_channels=610,kernel_size=3)

        self.linear1 = nn.Linear(480*21*21,245)
        
        self.linear2 = nn.Linear(245,128)
        
        self.linear3 = nn.Linear(128,80)
        
        self.linear4 = nn.Linear(80,output_dim)
        
    def forward(self,input):
        
        
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        maxpool1 = self.maxpool(conv2)
      

        conv3 = self.conv3(maxpool1)
        conv4 = self.conv4(conv3)
        maxpool2 = self.maxpool(conv4)

        conv5 = self.conv5(maxpool2)
        conv6 = self.conv6(conv5)
        maxpool3 = self.maxpoolLast(conv6)
        

        '''
        conv7 = self.conv7(maxpool3)
        conv8 = self.conv8(conv7)
        maxpool4 = self.maxpool(conv8)
        '''
        
        flatten = self.flatten(maxpool3)
        hidden_layer1 = self.linear1(flatten)
        
        hidden_layer2 = self.linear2(hidden_layer1)

        dropout1d = self.dropout(hidden_layer2)

        hidden_layer3 = self.linear3(dropout1d)
        
        output = self.linear4(hidden_layer3)
        return output
    
EPOCHS = 8
LEARNING_RATE = 0.001
INPUT_DIM = 3*112*112
OUTPUT_DIM = 43
STEPS = len(train_loader)
model = GTSRB_NETWORK(INPUT_DIM,OUTPUT_DIM).to(device)
optimizor = SGD(params=model.parameters(),lr=LEARNING_RATE)
loss = nn.CrossEntropyLoss()
try:
    for epoch in range(EPOCHS):
        with tqdm.trange(STEPS) as STEPS_:
            for step,(input,label) in enumerate(train_loader):
                input = input.to(device)
                label = label.to(device)
                prediction = model.forward(input)
                l = loss(prediction,label)
                l.backward()
                optimizor.step()
                optimizor.zero_grad()
                STEPS_.colour = 'green'
                STEPS_.desc = f'Epoch [{epoch}/{EPOCHS}], Step [{step}/{STEPS}], Loss [{l}]'
                STEPS_.update(1)
    torch.jit.script(model).save('./models/gtsrb_model.pt')
except KeyboardInterrupt:
    torch.jit.script(model).save('./models/gtsrb_model.pt')


for i,(input,label) in enumerate(train_loader):
    
    normalized_input = input[0].permute(1,2,0)
    prediction = model.forward(input.to(device)).argmax(1)[0]
    plt.title(f'predicted :{prediction} | ground truth : {label[0]}')
    plt.imshow(normalized_input)
    plt.show()
    
