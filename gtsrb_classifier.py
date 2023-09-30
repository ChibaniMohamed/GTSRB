import torch
import torch.nn as nn
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader,random_split
from torch.optim import Adam,lr_scheduler
from torchvision.transforms import ToTensor,Resize,Compose,ColorJitter,RandomRotation,Normalize
import matplotlib.pyplot as plt
import pickle
import tqdm
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
train_transforms = Compose([
    RandomRotation(15),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    Resize([50,50]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
validation_transforms =  Compose([
    Resize([50,50]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train_test_split(dataset,train_size):

    train_size = int(train_size * len(dataset))
    test_size = int(len(dataset) - train_size)
    return random_split(dataset,[train_size,test_size])


dataset = GTSRB(root='./gtsrb_dataset/',split="train")
train_set,validation_set = train_test_split(dataset,train_size=0.7)

train_set.dataset.transform = train_transforms
validation_set.dataset.transform = validation_transforms

print(f'training size : {len(train_set)}, Validation size : {len(validation_set)}')
train_loader = DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True)
validation_loader = DataLoader(dataset=validation_set,batch_size=BATCH_SIZE)






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

       

        self.l1 = nn.Linear(128*9*9,245)
        
        self.l2 = nn.Linear(245,128)
        self.l3 = nn.Linear(128,output_dim)
        
        
    def forward(self,input):
        
        conv = self.relu(self.conv1(input))
        conv = self.relu(self.conv2(conv))
        maxpool = self.maxpool1(conv)
        batchnorm = self.batchnorm1(maxpool)

        conv = self.relu(self.conv3(batchnorm))
        conv = self.relu(self.conv4(conv))
        maxpool = self.maxpool2(conv)
        batchnorm = self.batchnorm2(maxpool)

       

        flatten = self.flatten(batchnorm)
        
        dense_l1 = self.l1(flatten)
        dense_l2 = self.l2(dense_l1)
        output = self.l3(dense_l2)
        
       
        return output
    
    def training_metrics(self,positives,loss):
        data_size = len(train_loader)
        acc = positives/data_size
        return loss,acc
    
    def validation_metrics(self,validation_data,loss_function):
       data_size = len(validation_data)
       positives = 0
       val_loss = 0

       model = self.eval()
       with torch.no_grad() : 
        for step,(input,label) in enumerate(validation_data,start=1):
            input,label = input.to(device),label.to(device)
            prediction = model.forward(input)
            loss = loss_function(prediction,label)
            val_loss = loss.item()
            if prediction.argmax(1)[0] == label[0] :
                positives += 1
       
       val_acc = positives/data_size

       return val_loss,val_acc

    def save_metrics(self,output,filename,metrics_dict):

            
            with open(f"{output}/{filename}.pkl",'wb') as f:
                pickle.dump(metrics_dict,f)
                print('metrics saved !')


    def compile(self,train_data,validation_data,epochs,loss_function,optimizer,learning_rate_scheduler):
        val_acc_list = []
        val_loss_list = []

        train_acc_list = []
        train_loss_list = []

        learning_rate_list = []

        print('training started ...')
        STEPS = len(train_data)
        for epoch in range(epochs):
            lr = optimizer.param_groups[0]["lr"]
            learning_rate_list.append(lr)
            positives = 0
            loss = 0
            with tqdm.trange(STEPS) as STEPS_:

                for step,(input,label) in enumerate(train_loader,start=1):

                    input,label = input.to(device),label.to(device)
                    prediction = self.forward(input)

                    if prediction.argmax(1)[0] == label[0] :
                        positives += 1

                    l = loss_function(prediction,label)
                    loss = l.item()
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    STEPS_.colour = 'green'
                    STEPS_.desc = f'Epoch [{epoch}/{EPOCHS}], Step [{step}/{STEPS}], Learning Rate [{lr}], Loss [{"{:.4f}".format(l)}], Accuracy [{"{:.4f}".format(positives/step)}]'
                    STEPS_.update(1)

            training_loss,training_acc = self.training_metrics(positives,loss)
            train_acc_list.append(training_acc)
            train_loss_list.append(training_loss)

            val_loss, val_acc = self.validation_metrics(validation_data,loss_function)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            
            print(f'val_accuracy [{val_acc}], val_loss [{val_loss}]')

            
            learning_rate_scheduler.step()
        
        metrics_dict = {
                'train_acc':train_acc_list,
                'train_loss':train_loss_list,
                'val_acc':val_acc_list,
                'val_loss':val_loss_list,
                'learning_rate':optimizer.param_groups[0]["lr"]
            }
        self.save_metrics('./models','gtsrb_model_final_metrics',metrics_dict)
        print('training complete !')    

        
         

    
EPOCHS = 30
LEARNING_RATE = 0.001
INPUT_DIM = 3*50*50
OUTPUT_DIM = 43
model = GTSRB_NETWORK(INPUT_DIM,OUTPUT_DIM).to(device)
optimizer = Adam(params=model.parameters(),lr=LEARNING_RATE)
lr_s = lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.001,total_iters=10)
loss = nn.CrossEntropyLoss()
try:
    model.compile(train_data=train_loader,validation_data=validation_loader,epochs=EPOCHS,loss_function=loss,optimizer=optimizer,learning_rate_scheduler=lr_s)
    
    torch.jit.script(model).save('./models/gtsrb_model_final.pt')
except KeyboardInterrupt:
    torch.jit.script(model).save('./models/gtsrb_model_final.pt')



    
