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


class GoPro(Dataset):
    def __init__(self, data_root , mode="train", interFrames=1, n_inputs=2, ext="png"):

        super().__init__()
        self.interFrames = interFrames
        self.n_inputs = n_inputs
        self.setLength = (n_inputs-1)*(interFrames+1)+1 ## We require these many frames in total for interpolating `interFrames` number of
                                                ## intermediate frames with `n_input` input frames.
        self.data_root = data_root
        # video_list = os.listdir(self.data_root)
        with open('dataset/GoPro_test.txt') as t:
            video_list = t.readlines()
        self.frames_list = []
        self.file_list = []
        for video in video_list:
            video = video[:-1]
            if mode == "train":
                frames = sorted(os.listdir(os.path.join(self.data_root , video)))
            elif mode == 'test':
                frames = sorted(os.listdir(os.path.join(self.data_root , video, 'sharp')))
                
            n_sets = (len(frames) - self.setLength)//(interFrames+1)  + 1
            videoInputs = [frames[(interFrames+1)*i:(interFrames+1)*i+self.setLength] for i in range(n_sets)]
            if mode == "train":
                videoInputs = [[os.path.join(video, f) for f in group] for group in videoInputs]
            elif mode == 'test':
                videoInputs = [[os.path.join(video, 'sharp', f) for f in group] for group in videoInputs]
            self.file_list.extend(videoInputs)
        
        if mode == "train":
            self.transforms = transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
        elif mode == 'test':
            self.transforms = transforms.Compose([
                    transforms.ToTensor()
                ])

        print(len(self.file_list))

    def __getitem__(self, idx):

        imgpaths = [os.path.join(self.data_root , fp) for fp in self.file_list[idx]]
        if random.random() > 0.5:
            imgpaths = imgpaths[::-1] ## random temporal flip

        # We can use compression based augmentations

        pick_idxs = list(range(0,self.setLength,self.interFrames+1))
        rem = self.interFrames%2
        gt_idx = list(range(self.setLength//2-self.interFrames//2 , self.setLength//2+self.interFrames//2+rem)) 
        
        input_paths = [imgpaths[idx] for idx in pick_idxs]
        gt_paths = [imgpaths[idx] for idx in gt_idx]
       
        images = [Image.open(pth_) for pth_ in input_paths]
        images = [self.transforms(img_) for img_ in images]

        gt_images = [Image.open(pth_) for pth_ in gt_paths]
        gt_images = [self.transforms(img_) for img_ in gt_images] 
        
        h, w = images[0].shape[1], images[0].shape[2]
        
        coord0 = make_coord_3d((h, w), 0.)
        coord1 = make_coord_3d((h, w), 1 / 8)
        coord2 = make_coord_3d((h, w), 2 / 8)
        coord3 = make_coord_3d((h, w), 3 / 8)
        coord4 = make_coord_3d((h, w), 4 / 8)
        coord5 = make_coord_3d((h, w), 5 / 8)
        coord6 = make_coord_3d((h, w), 6 / 8)
        coord7 = make_coord_3d((h, w), 7 / 8)
        coord8 = make_coord_3d((h, w), 1.)
        # coords = [coord0, coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8]
        coords = [coord1, coord2, coord3, coord4, coord5, coord6, coord7]
        cell0, cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8 = torch.ones_like(coord1), torch.ones_like(coord1), torch.ones_like(coord1), torch.ones_like(coord1), torch.ones_like(coord1), torch.ones_like(coord1), torch.ones_like(coord1), torch.ones_like(coord1), torch.ones_like(coord1)
        cell0[:, 0] *= 2 / h
        cell0[:, 1] *= 2 / w
        cell1[:, 0] *= 2 / h
        cell1[:, 1] *= 2 / w
        cell2[:, 0] *= 2 / h
        cell2[:, 1] *= 2 / w
        cell3[:, 0] *= 2 / h
        cell3[:, 1] *= 2 / w
        cell4[:, 0] *= 2 / h
        cell4[:, 1] *= 2 / w
        cell5[:, 0] *= 2 / h
        cell5[:, 1] *= 2 / w
        cell6[:, 0] *= 2 / h
        cell6[:, 1] *= 2 / w
        cell7[:, 0] *= 2 / h
        cell7[:, 1] *= 2 / w
        cell8[:, 0] *= 2 / h
        cell8[:, 1] *= 2 / w
        
        cell0[:, 2] *= 0.
        cell1[:, 2] *= 1 / 8
        cell2[:, 2] *= 2 / 8 
        cell3[:, 2] *= 3 / 8 
        cell4[:, 2] *= 4 / 8 
        cell5[:, 2] *= 5 / 8 
        cell6[:, 2] *= 6 / 8
        cell7[:, 2] *= 7 / 8 
        cell8[:, 2] *= 1.
        # cells = [cell0, cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8]
        cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7]
        # time_stamps = [0., 1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1.]
        time_stamps = [1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8]
        
        return images, gt_images, coords, cells, time_stamps
        
        # return images , gt_images
        

    def __len__(self):

        return len(self.file_list)

def get_loader(mode, data_root, batch_size, shuffle, num_workers, interFrames=7, n_inputs=4, random_seed=0):

    # if test_mode:
    #     mode = "test"
    # else:
    #     mode = "train"
    
    dataset = GoPro(data_root, mode, interFrames=interFrames, n_inputs=n_inputs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)



def make_coord_3d(shape, time, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
        bias = torch.tensor(time).repeat(ret.shape[0], 1)
        ret = torch.cat([ret, bias], dim=1)
    return ret


def make_coord_2d(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


if __name__ == "__main__":

    dataset = GoPro("./GoPro" , mode="train", interFrames=5, n_inputs=2)

    print(len(dataset))

    dataloader = DataLoader(dataset , batch_size=1, shuffle=True, num_workers=0)
