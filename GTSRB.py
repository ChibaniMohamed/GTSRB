from torch.utils.data import Dataset
from typing import Any,Tuple,Optional,Callable
import PIL
import csv
import pathlib


class GTSRB(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None):
       
        
        
        self.base_folder = pathlib.Path(root)
        self.csv_file = self.base_folder / ('train.csv' if split =='train' else 'test.csv')
        

        with open(self.csv_file) as csvfile:
           samples = [(str(self.base_folder / row['Path']),int(row['ClassId'])) 
            for row in csv.DictReader(csvfile,delimiter=',',skipinitialspace=True)
                ]


        self.samples = samples
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple:
        path,classId =  self.samples[index]
        sample = PIL.Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample,classId
     





