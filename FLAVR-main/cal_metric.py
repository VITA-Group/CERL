import os
import sys
import time
import copy
import shutil
import random
import pdb

import torch
import numpy as np
from tqdm import tqdm

import config
import myutils
import models

import torch.nn as nn
from torch.utils.data import DataLoader
from pdb import set_trace as bp
from PIL import Image

from gpu_memory_log import gpu_memory_log

##### Parse CmdLine Arguments #####
# os.environ["CUDA_VISIBLE_DEVICES"]='7'
args, unparsed = config.get_args()
cwd = os.getcwd()

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

if args.dataset == "vimeo90K_septuplet":
    from dataset.vimeo90k_septuplet import get_loader
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "ucf101":
    from dataset.ucf101_test import get_loader
    test_loader = get_loader(args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "gopro":
    from dataset.GoPro import get_loader
    test_loader = get_loader('train', '/data/nnice1216/high_FPS_video/GOPRO_Large_all/', 1, shuffle=False, num_workers=8, interFrames=7)  
    print("Dataset Prepared!")
else:
    raise NotImplementedError

##### FLAVR #####
'''
from model.FLAVR_arch import UNet_3D_3D
print("Building model: %s"%args.model.lower())
model = UNet_3D_3D(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType)

model = torch.nn.DataParallel(model).to(device)
model_dict = model.state_dict()
model.load_state_dict(torch.load(args.load_from)["state_dict"] , strict=True)
print("#params" , sum([p.numel() for p in model.parameters()]))
'''
##### LIIF #####
model_args = {'encoder_spec': {'name': 'edsr-baseline', 'args': {'no_upsampling': True}}, 'imnet_spec': {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [64, 64]}}}
model_spec = {'name': 'liif_bidi', 'args': model_args}
model = models.make(model_spec).to(device)
model = nn.DataParallel(model, device_ids=device_ids)
name = 'final-739.pth'
model.load_state_dict(torch.load('/model/nnice1216/video/' + name))


def test(args):
    time_taken = []
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    out_dir = "/output/gopro_test/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    psnr_list = []
    with torch.no_grad():
        for i, (images, gt_image, coords, cells, times) in enumerate(tqdm(test_loader)):
            # if i > 100:
            #     break
            images = [img_[None].cuda() for img_ in images]
            images = torch.cat(images, dim=0).permute(1, 2, 0, 3, 4)
            gt = [g_.cuda() for g_ in gt_image]
            coords = [c_.cuda() for c_ in coords]
            cells = [c_.cuda() for c_ in cells]

            torch.cuda.synchronize()
            start_time = time.time()
            # out = model(images)
        
            out = model(images, coords, "testing")
            
            out = torch.cat(out)
            gt = torch.cat(gt)
            '''
            for j in range(7):
                Image.fromarray((gt[j].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, 'Epoch{}_iter{}_GT.jpg'.format(i, j)))
                Image.fromarray((out[j].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, 'Epoch{}_iter{}_PRED.jpg'.format(i, j)))
            '''
            torch.cuda.synchronize()
            time_taken.append(time.time() - start_time)
            myutils.eval_metrics(out, gt, psnrs, ssims)

            print("PSNR: %f, SSIM: %f" %
          (psnrs.avg, ssims.avg))
    print("FINAL: PSNR: %f, SSIM: %f" %
          (psnrs.avg, ssims.avg))
    print("Time: " , sum(time_taken)/len(time_taken))

    return psnrs.avg


""" Entry Point """
def main(args):
    
    assert args.load_from is not None

    test(args)


if __name__ == "__main__":
    main(args)