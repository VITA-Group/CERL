import os
import sys
import time
import argparse
from PIL import Image
# from generate_video import generate_video

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torchvision.models import vgg16
from perceptual import LossNetwork
import random

import config
import myutils
from myutils import test_images_3d, test_images_2d, test_images_outp, test_metric, adjust_learning_rate, convert_to_gray, get_raft_args
from loss import Loss
from loss import edge_conv2d
from torch.utils.data import DataLoader
from loss import L1_Charbonnier_loss
from pdb import set_trace as bp

import models
from utils import make_coord
from torch.autograd import Variable
from loss import PerceptualLoss, DCLoss

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

import cv2
import warnings

from dataset.GoPro_arbitrary_nosr import make_coord_3d


def double_forward(model, optimizer, preds, images, gt, device, i, epoch_id, bs):
    
    sampled_idx = sorted(random.sample(range(5), 3))
    h, w = preds[0].shape[2], preds[0].shape[3]
    for idx in range(3):
        optimizer.zero_grad()
        temp_coord = make_coord_3d((h, w), (idx + 2) / 8)
        temp_coord = [temp_coord.to(device)[None].repeat(bs, 1, 1)]
        if idx == 0:
            inputs = torch.cat([images[:, :, 1].unsqueeze(0), preds[1].detach().unsqueeze(0), preds[3].detach().unsqueeze(0), preds[5].detach().unsqueeze(0)], dim=0).permute(1, 2, 0, 3, 4)
        elif idx == 1:
            inputs = torch.cat([preds[0].detach().unsqueeze(0), preds[2].detach().unsqueeze(0), preds[4].detach().unsqueeze(0), preds[6].detach().unsqueeze(0)], dim=0).permute(1, 2, 0, 3, 4)
        else:
            inputs = torch.cat([preds[1].detach().unsqueeze(0), preds[3].detach().unsqueeze(0), preds[5].detach().unsqueeze(0), images[:, :, 2].unsqueeze(0)], dim=0).permute(1, 2, 0, 3, 4)
        new_pred_f, new_pred_b = model(inputs, temp_coord, True)
        loss = F.smooth_l1_loss(new_pred_f[0], gt[idx + 2])\
            + F.smooth_l1_loss(new_pred_b[0], gt[idx + 2])\
            + F.smooth_l1_loss(new_pred_b[0], new_pred_f[0])
        loss.backward()
        optimizer.step()
        print('Epoch %d, Iter %d, Loss: %.4f' % (epoch_id, i, loss.item()))
        
    return 
warnings.filterwarnings('ignore')

##### Tensorboard #####
writer = SummaryWriter('/output/logs')

##### Parameters #####
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--epoch_num', type=int, default=30)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--if_continue', type=bool, default=False)
parser.add_argument('--TEMP', type=float, default=1)

args = parser.parse_known_args()[0]

##### Dataset ###### 
## DAVIS 
# from dataset.Davis_liif import get_loader
# args.data_root = '/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/' 
# train_loader = get_loader(args.data_root, args.batch_size, shuffle=True, num_workers=8, drop_last=True)


## VIMEO ##
# from dataset.vimeo90k_septuplet import get_loader
# args.data_root = '/data/nnice1216/vimeo_septuplet/'
# train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)


## GOPRO ##
from dataset.GoPro_arbitrary_nosr import get_loader
random_seed = 0
interval = 5
# train_data_root = '/data/nnice1216/X4K1000FPS_dataset/train/'
train_data_root = '/data/nnice1216/high_FPS_video/GOPRO_Large_all/'
# train_data_root = '/data/nnice1216/vimeo_septuplet/'
train_loader = get_loader('train', train_data_root, args.batch_size, shuffle=True, num_workers=args.num_workers, random_seed=random_seed, seted_interval=interval)
# train_loader = get_loader('train', train_data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)


##### Model #####
## LIIF ###
model_args = {'encoder_spec': {'name': 'edsr-baseline', 'args': {'no_upsampling': True}}, 'imnet_spec': {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [64, 64]}}}
model_spec = {'name': 'liif_gtoptical', 'args': model_args}
model = models.make(model_spec).to(device)
model = nn.DataParallel(model, device_ids=device_ids)
# for k, v in model.named_parameters():
#     if k[:14] == 'module.encoder':
#         v.requires_grad=False
        
# model.load_state_dict(torch.load('/model/nnice1216/video/FLAVR_2x.pth')['state_dict'], strict=False)

if args.if_continue:
    name = 'temp-153.pth'
#     name = 'vimeo_epoch1_iter499.pth'
    print('Load model ' + name)
    model.load_state_dict(torch.load('/model/nnice1216/video/' + name))

##### Loss & Optimizer #####
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

##### Training #####
model.train()
loss_f = PerceptualLoss(nn.MSELoss(reduce=True))
index = 0

args1 = get_raft_args()
raft_model = RAFT(args1).to(device)
raft_model = nn.DataParallel(raft_model, device_ids=device_ids)
raft_model.load_state_dict(torch.load(args1.model))
raft_model.eval()
        
        
for epoch_id in range(args.epoch_num):
    
    print('Epoch {} Begin'.format(epoch_id))
    lr = adjust_learning_rate(epoch_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = '/output/models/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir2 = '/output/tempimgs22/'
    if not os.path.exists(out_dir2):
        os.makedirs(out_dir2)

    for i, data in enumerate(train_loader):
        
        ## VIMEO PRE_PROCESS FOR LIIF_3D ##
        # images, gt_image, coords, cells, times = data
        images, gt_image, coords, _, _ = data
        images = [img_.to(device) for img_ in images]
        images = torch.stack(images, dim=2)
        gt = [g_.to(device) for g_ in gt_image]
        # gt = [g_.view(args.batch_size, 3, -1).permute(0, 2, 1).to(device) for g_ in gt_image]
        # gt = torch.cat(gt).view(args.batch_size, 3, -1).permute(0, 2, 1)
        coords = [c_.to(device) for c_ in coords]
        # coord, cell = coord.to(device), cell.to(device)
 
        # Forward
        optimizer.zero_grad()
        # pred_inter, pred_0, pred_1 = model(img, coord, cell)
        # bp()
        
        preds_f, masks, flows = model(images, coords, True, gts=gt)
        
        loss1 = 0
        loss2 = 0
        loss3 = 0
        loss4 = 0
        
        for idx in range(3):
#             diff = torch.abs(warps[idx] - convert_to_gray(gt[idx]))
#             diff = (diff - diff.min()) / (diff.max() - diff.min())
            loss1 += F.smooth_l1_loss(preds_f[idx], gt[idx])
            loss4 += F.smooth_l1_loss(masks[idx], flows[idx])
            # loss1 += (F.smooth_l1_loss(preds_f[idx], gt[idx], reduce=False) * (att_map1[idx].unsqueeze(1) + 1)).mean()
            loss2 += loss_f.get_loss(preds_f[idx], gt[idx]) * 0.05
            # loss3 += loss_f.get_loss(preds_f[idx] * ((att_map1[idx].unsqueeze(1)) / 2), gt[idx] * ((att_map1[idx].unsqueeze(1)) / 2)) * 0.05
        
        # Backward & Update
        loss = loss1 + loss2 + loss3 + loss4
        loss.backward()
        optimizer.step()
        
        print('Epoch %d, Iter %d, Loss: %.4f' % (epoch_id, i, loss.item()))
        print('Loss1: %.4f, Loss2: %.4f, Loss4: %.4f' % (loss1.item(), loss2.item(), loss4.item()))
        # print('Loss1: %.4f, Loss2: %.4f, Loss3: %.4f' % (loss1.item(), loss2.item(), loss3.item()))
        # print('Epoch %d, Iter %d, Loss1: %.4f, Loss2: %.4f, Loss: %.4f' % (epoch_id, i, loss1.item(), loss2.item(), loss.item()))
        writer.add_scalar('Training Loss', loss.item(), index)
        writer.add_scalar('L1 Loss', loss1.item(), index)
        # writer.add_scalar('Perceptual Loss', loss2.item(), index)
        index += 1
        # index += 1
        
        if i % 25 == 0:
            Image.fromarray((gt_image[0][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir2, 'Epoch{}_iter{}_GT.jpg'.format(epoch_id, i)))
            Image.fromarray((preds_f[0][0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir2, 'Epoch{}_iter{}_PRED.jpg'.format(epoch_id, i)))
            Image.fromarray((flows[0][0, 0].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir2, 'Epoch{}_iter{}_MASKGT.jpg'.format(epoch_id, i)))
            Image.fromarray((masks[0][0, 0].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir2, 'Epoch{}_iter{}_MASK.jpg'.format(epoch_id, i)))
#             Image.fromarray((masks[0][0][0].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir2, 'Epoch{}_iter{}_MASK.jpg'.format(epoch_id, i)))
#             diff = torch.abs(warps[idx] - convert_to_gray(gt[idx]))
#             diff = (diff - diff.min()) / (diff.max() - diff.min())
#             img = (diff.clamp(0, 1).detach().cpu() * 255)[0, 0].numpy().astype(np.uint8)
#             Image.fromarray(img).save(os.path.join(out_dir2, 'Epoch{}_iter{}_MASK_GT.jpg'.format(epoch_id, i)))
            
        # if i == 50:
        #     test_metric(model, epoch_id, i, True)
        
        if i % 100 == 0:
            test_images_outp(model, device, 1 / 64, epoch_id, i)
        if i % 100 == 10:
            test_images_outp(model, device, 11 / 64, epoch_id, i)
        if i % 100 == 20:
            test_images_outp(model, device, 23 / 64, epoch_id, i)
        elif i % 100 == 40:
            test_images_outp(model, device, 32 / 64, epoch_id, i)
        elif i % 100 == 60:
            test_images_outp(model, device, 43 / 64, epoch_id, i)
        elif i % 100 == 80:
            test_images_outp(model, device, 52 / 64, epoch_id, i)
        elif i % 100 == 90:
            test_images_outp(model, device, 63 / 64, epoch_id, i)
            # out_path = '/output/Image22s/Epoch_{}'.format(epoch_id)
            # Image.open('/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/00003.jpg').resize((720, 416)).save(os.path.join(out_path, 'Iter{}.jpg'.format(i)))
            # test_images_outp(model, device, 7 / 8, epoch_id, i)
            # test_images_3d(model, device, 7 / 8, epoch_id, i)
            
        model.train()
        # double_forward(model, optimizer, preds_f, images, gt, device, i, epoch_id, args.batch_size)
        
        if (i + 1) % 100 == 0:
            torch.save(model.state_dict(), '/output/models/vimeo_epoch{}_iter{}.pth'.format(epoch_id, i))
        if (i + 1) % 1000 == 0:
            test_metric(model, epoch_id, i)
    # if epoch_id % 7 == 6:
    #     test_metric(model, epoch_id, i)
    print("Epoch {} Done. Index={}".format(epoch_id, index))