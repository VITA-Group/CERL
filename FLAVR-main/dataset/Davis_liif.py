import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob
from pdb import set_trace as bp


class Davis_liif(Dataset):
    def __init__(self, data_root, ext="png"):

        super().__init__()

        self.data_root = data_root
        self.name_list = os.listdir(data_root)

        self.transforms = transforms.Compose([
                transforms.RandomCrop((256,256)),
                transforms.ToTensor()
            ])

        print(len(self.name_list))

    def __getitem__(self, idx):
    
        imgpath = os.path.join(self.data_root, self.name_list[idx])
        image = Image.open(imgpath)
        image = self.transforms(image)
        
        h, w = image.shape[1], image.shape[2]
        coord = make_coord((h, w))
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        
        return image, coord, cell

    def __len__(self):

        return len(self.name_list)

    
def get_loader(data_root, batch_size, shuffle, num_workers, test_mode=True, drop_last=True):

    dataset = Davis_liif(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)


def make_coord(shape, ranges=None, flatten=True):
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

    dataset = Davis_liif("./Davis_test/")

    print(len(dataset))

    dataloader = DataLoader(dataset , batch_size=1, shuffle=True, num_workers=0)