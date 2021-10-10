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
    def __init__(self, data_root , mode="train", interFrames=1, n_inputs=2, ext="png", random_seed=0, seted_interval=4):

        super().__init__()

        self.interFrames = interFrames
        self.n_inputs = n_inputs
        self.setLength = (n_inputs-1)*(interFrames+1)+1 ## We require these many frames in total for interpolating `interFrames` number of
                                                ## intermediate frames with `n_input` input frames.
        self.data_root = data_root
        video_list = os.listdir(self.data_root)
        self.frames_list = []

        self.file_list = []
        self.gt_list = []
        
        for video in video_list:
            sharp_path = os.path.join(self.data_root , video, 'sharp')
            blur_path = os.path.join(self.data_root , video, 'blur')
            # interval = 7
            sharp_frames = sorted(os.listdir(sharp_path))
            blur_frames = sorted(os.listdir(blur_path))
            
            temp_sharp_list = [os.path.join(sharp_path, frame_name) for frame_name in sharp_frames]
            temp_blur_list = [os.path.join(blur_path, frame_name) for frame_name in blur_frames]
            
            self.gt_list.extend(temp_sharp_list)
            self.file_list.extend(temp_blur_list)
            
        self.transforms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        
        print(len(self.file_list))
        print(len(self.gt_list))

    def __getitem__(self, idx):
                
        gt_path = self.gt_list[idx]
        blur_path = self.file_list[idx]
        
        gt_sampled_idx = sorted(random.sample(range(7), 3))
            
        iinterFrames = 7
        ssetLength = (self.n_inputs - 1) * (iinterFrames + 1) + 1
       
        image = Image.open(blur_path)
        image = self.transforms(image)
        gt_image = Image.open(gt_path)
        gt_image = self.transforms(gt_image)
        
        h, w = image.shape[1], image.shape[2]
        
        # coord0 = make_coord_3d((768, 768), 0.)
        # cell0 = torch.ones_like(coord0)
        # cell0[:, 0] *= 2 / 768
        # cell0[:, 1] *= 2 / 768
        # cell0[:, 2] *= 0.
        # coords = [coord0]
        # cells = [cell0]
        # time_stamps = [0.]
        
        coords = []
        cells = []
        time_stamps = []
        for i in gt_sampled_idx:
            temp_coord = make_coord_3d((h, w), (i) / (iinterFrames + 1))
            temp_cell = torch.ones_like(temp_coord)
            temp_cell[:, 0] *= 2 / h
            temp_cell[:, 1] *= 2 / w
            temp_cell[:, 2] *= (i) / (iinterFrames + 1)
            # temp_cell[:, 3] *= iinterFrames
            coords.append(temp_coord)
            cells.append(temp_cell)
            time_stamps.append((i) / (iinterFrames + 1))
        # time_stamps.append(1.)
        # coord1 = make_coord_3d((768, 768), 1.)
        # cell1 = torch.ones_like(coord0)
        # cell1[:, 0] *= 2 / 768
        # cell1[:, 1] *= 2 / 768
        # cell1[:, 2] *= 1.
        # coords.append(coord1)
        # cells.append(cell1)
        return image, gt_image, coords, cells, time_stamps
        

    def __len__(self):

        return len(self.file_list)

def get_loader(mode, data_root, batch_size, shuffle, num_workers, random_seed, seted_interval, interFrames=7, n_inputs=4):

    # if test_mode:
    #     mode = "test"
    # else:
    #     mode = "train"

    dataset = GoPro(data_root , mode, interFrames=interFrames, n_inputs=n_inputs, random_seed=random_seed, seted_interval=seted_interval)
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


def make_coord_4d(shape, time, interval, ranges=None, flatten=True):
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
        bias2 = torch.tensor(interval).repeat(ret.shape[0], 1)
        ret = torch.cat([ret, bias2, bias], dim=1)
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
