import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob
import pdb
from pdb import set_trace as bp


class TEMP(Dataset):
    def __init__(self):

        super().__init__()
        
        self.path1 = '/data/nnice1216/high_FPS_video/GOPRO_Large_all/GOPR0384_11_00/'
        self.path2 = '/output/Imagestry11/'
        
        self.file_list = sorted(os.listdir(self.path2))[2:-4]
        print(self.file_list)
        print(len(self.file_list))
        self.totensor = transforms.ToTensor()
        
    
    def __getitem__(self, idx):
        
        file_name = self.file_list[idx][:-4]
        gt_path = os.path.join(self.path1, file_name + '.png')
        pred_path = os.path.join(self.path2, file_name + '.jpg')
        
        gt_img = Image.open(gt_path)
        pred_img = Image.open(pred_path)
        
        gt_img = self.totensor(gt_img)
        pred_img = self.totensor(pred_img)
        
        return gt_img, pred_img
        

    def __len__(self):

        return len(self.file_list)