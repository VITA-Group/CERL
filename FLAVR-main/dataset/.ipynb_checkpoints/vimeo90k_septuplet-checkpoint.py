  
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class VimeoSepTuplet(Dataset):
    def __init__(self, data_root, is_training , input_frames="1357"):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training
        self.inputs = input_frames

        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()

        self.transforms = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor()
        ])
            
    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        
        imgpaths = [imgpath + f'/im{i}.png' for i in range(1,8)]
        pth_ = imgpaths

        # Load images
        images = [Image.open(pth) for pth in imgpaths]

        ## Select only relevant inputs
        inputs = [int(e)-1 for e in list(self.inputs)]
        inputs = inputs[:len(inputs)//2] + [3] + inputs[len(inputs)//2:]
        images = [images[i] for i in inputs]
        imgpaths = [imgpaths[i] for i in inputs]
        # Data augmentation
        seed = random.randint(0, 2**32)
        images_ = []
        for img_ in images:
            random.seed(seed)
            images_.append(self.transforms(img_))
        images = images_
        
        # Random Temporal Flip
        if random.random() >= 0.5:
            images = images[::-1]
            imgpaths = imgpaths[::-1]

        gt = images[len(images)//2]
        images = images[:len(images)//2] + images[len(images)//2+1:]
        
        h, w = images[0].shape[1], images[0].shape[2]
        coord = make_coord_3d((h, w), 1 / 2)
        
        return images, [gt], [coord]

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)

def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoSepTuplet(data_root, is_training=is_training)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

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


if __name__ == "__main__":

    dataset = VimeoSepTuplet("./vimeo_septuplet/", is_training=True)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=32, pin_memory=True)