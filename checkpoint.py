import torch
import torch.nn as nn
from torchvision.datasets import GTSRB
from torchvision.transforms import Resize,ToTensor,Compose
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam
import tqdm

transforms = Compose([
    Resize([112,112]),
    ToTensor()
])
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = GTSRB(root='./gtsrb_dataset/',split="train",transform=transforms)
train_loader = DataLoader(dataset=train_dataset,shuffle=True)



EPOCHS = 8
LEARNING_RATE = 0.001
INPUT_DIM = 3*112*112
OUTPUT_DIM = 43
STEPS = len(train_loader)
model = torch.jit.load('./models/gtsrb_model.pt').to(device)
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