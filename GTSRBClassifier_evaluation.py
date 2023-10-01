import torch
from GTSRB import GTSRB
from torchvision.transforms import Resize,ToTensor,Compose,Normalize,RandomAutocontrast,RandomRotation,GaussianBlur,ColorJitter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

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
  
  with torch.no_grad() : 
    for id,(input,label) in enumerate(iter(test_dataloader),start=1):
        input = input.to(device)
        label = label.to(device)
        prediction = gtsrbClassifier.forward(input).argmax(1).cpu().numpy()[0]
        label = label.cpu().numpy()
        if label == prediction:
            positives += 1
        '''
        else:
           input = input[0].permute(1,2,0).cpu()
           plt.title(f'prediction : {prediction}, truth : {label[0]}')
           plt.imshow(input)
           plt.show()
        '''
        progress.update(1)
        progress.desc = f"Accuracy : {positives/id}, Positives : {positives}, Negatives : {id-positives}"
       

'''
    input = input.cpu()
    plt.title(f'prediction : {prediction} | ground truth : {label[0]}')
    plt.imshow(input[0].permute(1,2,0))
    plt.show()
    '''