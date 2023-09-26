import torch
from torchvision.datasets import GTSRB
from torchvision.transforms import Resize,ToTensor,Compose
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

transforms = Compose([
    Resize([112,112]),
    ToTensor()
])
device = "cuda" if torch.cuda.is_available() else "cpu"

testdata = GTSRB(root='./gtsrb_dataset/',split='test',transform=transforms)

test_dataloader = DataLoader(testdata,shuffle=True)

gtsrbClassifier = torch.jit.load('./models/gtsrb_model.pt').eval().to(device)

with tqdm(colour='red',total=len(test_dataloader)) as progress:
  positives = 0
  
  with torch.no_grad() : 
    for id,(input,label) in enumerate(iter(test_dataloader),start=1):
        input = input.to(device)
        label = label.to(device)
        prediction = gtsrbClassifier.forward(input).argmax(1).cpu().numpy()[0]
        label = label.cpu().numpy()
        if label == prediction:
            positives += 1
        progress.update(1)
        progress.desc = f"Accuracy : {positives/id}, Positives : {positives}, Negatives : {id-positives}"
       

'''
    input = input.cpu()
    plt.title(f'prediction : {prediction} | ground truth : {label[0]}')
    plt.imshow(input[0].permute(1,2,0))
    plt.show()
    '''