import os
import sys
import time
import argparse
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import config
import myutils
from myutils import test_images_3d, test_images_2d, test_images_outp
from loss import Loss
from torch.utils.data import DataLoader
from pdb import set_trace as bp

import models
from utils import make_coord
from torch.autograd import Variable

# from dataset.Davis_test import get_loader

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


def get_name(index):
    if index >= 0 and index <= 9:
        text = '0000' + str(index) + '.jpg'
    elif index >= 10 and index <= 99:
        text = '000' + str(index) + '.jpg'
    elif index >= 100 and index <= 999:
        text = '00' + str(index) + '.jpg'
    else:
        text = '0' + str(index) + '.jpg'
    return text


def save_image(img, index):
    if index < 1600:
        img.save(os.path.join(out_path1, get_name(index)))
    elif index < 3200:
        img.save(os.path.join(out_path2, get_name(index)))
    elif index < 4800:
        img.save(os.path.join(out_path3, get_name(index)))
    elif index < 6400:
        img.save(os.path.join(out_path4, get_name(index)))
    elif index < 8000:
        img.save(os.path.join(out_path5, get_name(index)))
    elif index < 9600:
        img.save(os.path.join(out_path6, get_name(index)))
    else:
        img.save(os.path.join(out_path7, get_name(index)))
        
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_args = {'encoder_spec': {'name': 'edsr-baseline', 'args': {'no_upsampling': True}}, 'imnet_spec': {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [64, 64]}}}
model_spec = {'name': 'liif_optical', 'args': model_args}
model = models.make(model_spec).to(device)
model = nn.DataParallel(model, device_ids=device_ids)
name = 'temp-959-0-99.pth'
model.load_state_dict(torch.load('/model/nnice1216/video/' + name))

# imgpath = '/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/insects-original/'
imgpath = '/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/insects_sampled/'
model.eval()
epoch = 0
# time = 0.5
iter_id = 1
signal = 0
with torch.no_grad():
    name_list = sorted(os.listdir(imgpath))
    # name_list = sorted(name_list)[20:140:2]
    index = 0
    out_path1 = '/output/Imagestry11/'
    out_path2 = '/output/Imagestry22/'
    out_path3 = '/output/Imagestry33/'
    out_path4 = '/output/Imagestry44/'
    out_path5 = '/output/Imagestry55/'
    out_path6 = '/output/Imagestry66/'
    out_path7 = '/output/Imagestry77/'
    if not os.path.exists(out_path1):
        os.makedirs(out_path1)
    if not os.path.exists(out_path2):
        os.makedirs(out_path2)
    if not os.path.exists(out_path3):
        os.makedirs(out_path3)
    if not os.path.exists(out_path4):
        os.makedirs(out_path4)
    if not os.path.exists(out_path5):
        os.makedirs(out_path5)
    if not os.path.exists(out_path6):
        os.makedirs(out_path6)
    if not os.path.exists(out_path7):
        os.makedirs(out_path7)
        
            
    for image_id in range(len(name_list) - 3):

        start = time.time()
        imgpaths = [name_list[i] for i in range(image_id, image_id + 4)]
        pth_ = imgpaths

        images = [Image.open(os.path.join(imgpath, pth)) for pth in imgpaths]
        h, w = images[0].size
        # print(h, w)
        images = [img.resize((960, 544)) for img in images]
        save_image(images[1], index)
        print("INDEX {}, DONE!".format(index))
        # images[1].save(os.path.join(out_path, get_name(index)))
        index += 1
        print("START: ", image_id, index)

        T = transforms.ToTensor()
        # images = [((T(img_) - 0.5) * 2)[None] for img_ in images]
        images = [T(img_)[None] for img_ in images]
        h, w = images[0].shape[2], images[0].shape[3]
        print(h, w)
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

        coords = [c_[None].to(device) for c_ in coords]
        cells = [ce_[None].to(device) for ce_ in cells]

        images = [img_.to(device) for img_ in images]
        images = torch.stack(images, dim=2)
        
        torch.cuda.synchronize()
        
        out = model(images, coords, 8, index)

        torch.cuda.synchronize()
        index += 62
        # for out in outs:
        out_temp = (out).clamp(0, 1).permute(0, 2, 3, 1)[0].cpu()
        save_image(Image.fromarray((out_temp.numpy() * 255).astype(np.uint8)), index)
        print("INDEX {}, DONE!".format(index))
        index += 1
        end = time.time()
        print("Epoch {} End, Index {}, Cost time {}".format(image_id, index, end - start))
        
        del out, out_temp
        torch.cuda.empty_cache() 
        if index > 1600 and signal == 0: 
            fps = 60
            size = (1280, 720)
            path = '/output/Imagestry11'
            video = cv2.VideoWriter("/output/Video2.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

            for i, name in enumerate(sorted(os.listdir(path))):
                if i % 50 == 0:
                    print(name)
                img = cv2.imread(os.path.join(path, name))
                video.write(img)

            video.release()
            signal = 1