import torch
from GTSRB import GTSRB
from torchvision.transforms import Resize,ToTensor,Compose,RandomVerticalFlip,RandomHorizontalFlip,RandomRotation,GaussianBlur,ColorJitter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np

transforms = Compose([
    Resize([50,50]),
    ToTensor(),
    
])
device = "cuda" if torch.cuda.is_available() else "cpu"

testdata = GTSRB(root='./archive',split='test',transform=transforms)
print('testing size :',len(testdata))
test_dataloader = DataLoader(testdata)


gtsrbClassifier = torch.jit.load('./models/gtsrb_model_final.pt').eval().to(device)

with tqdm(colour='red',total=len(test_dataloader)) as progress:
  positives = 0
  total = 0
  with torch.no_grad() : 
    for id,(input,label) in enumerate(iter(test_dataloader)):
        input = input.to(device)
        label = label.to(device)
        prediction = gtsrbClassifier.forward(input)
        _,predicted = torch.max(prediction,1)
        positives += (predicted == label).sum().item()
        total += 1
        
        if predicted != label :
           input = ((input.cpu().squeeze().permute(1,2,0).numpy() + 1 )/2) * 255
           Image.fromarray(input.astype(np.uint8)).save(f'./false_prediction/{label.item()}-{id}.jpg')
        
        progress.update(1)
        progress.desc = f"Accuracy : {positives/total}, Positives : {positives}, Negatives : {id-positives}"
       


  
   