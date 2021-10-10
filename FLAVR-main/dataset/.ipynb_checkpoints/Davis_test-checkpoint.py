import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import glob


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




class Davis(Dataset):
    def __init__(self, data_root , ext="png"):

        super().__init__()

        self.data_root = data_root
        self.name_list = os.listdir(data_root)

        self.T = transforms.ToTensor()

        print(len(self.name_list))

    def __getitem__(self, idx):

        imgpaths = [self.name_list[i] for i in range(idx, idx + 4)]
        images = [Image.open(os.path.join(self.data_root, pth)) for pth in imgpaths]
        images = [self.T(img.resize((960, 544))).clone() for img in images]
        # images = [self.T(img_)[None] for img_ in images]

        h, w = images[0].shape[1], images[0].shape[2]
        coord0 = make_coord_3d((h, w), 0.0)
        coord1 = make_coord_3d((h, w), 1 / 8)
        coord2 = make_coord_3d((h, w), 2 / 8)
        coord3 = make_coord_3d((h, w), 3 / 8)
        coord4 = make_coord_3d((h, w), 4 / 8)
        coord5 = make_coord_3d((h, w), 5 / 8)
        coord6 = make_coord_3d((h, w), 6 / 8)
        coord7 = make_coord_3d((h, w), 7 / 8)
        coords = [coord0, coord1, coord2, coord3, coord4, coord5, coord6, coord7]
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
        cells = [cell0, cell1, cell2, cell3, cell4, cell5, cell6, cell7]

        return images, coords, cells, idx

    def __len__(self):

        return len(self.name_list) - 3

def get_loader(data_root, batch_size, shuffle, num_workers):

    dataset = Davis(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":

    dataset = Davis("./Davis_test/")

    print(len(dataset))

    dataloader = DataLoader(dataset , batch_size=1, shuffle=True, num_workers=0)