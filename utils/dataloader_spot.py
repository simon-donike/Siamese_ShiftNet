from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch


class dataset_spot6(Dataset):
    def __init__(self,folder_path):
        self.folder_path = folder_path
        # list files in data folder and filter tifs
        self.files = os.listdir(self.folder_path)
        self.files = [string for string in self.files if string.endswith(".tif")]
        
    def __len__(self):
        # return len of dataset
        return(len(self.files))
    
    def __getitem__(self,idx):
        # open, manipulate and return image
        file = self.files[idx] # get according file by index
        im = Image.open(self.folder_path+file) # read file
        im = np.array(im) # turn to np array
        im = im/1000 # get datarange to 0...1
        im = im.transpose(2,0,1) # reshape to C,W,H
        im = torch.from_numpy(im) # turn to numpy
        im = im.float() # set as float to match with model dtype
        return(im)  # return np array file