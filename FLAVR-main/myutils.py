# from https://github.com/myungsub/CAIN/blob/master/utils.py, 
# but removed the errenous normalization and quantization steps from computing the PSNR.

from pytorch_msssim import ssim_matlab as calc_ssim
import math
import os
import torch
import shutil
from PIL import Image
from utils import make_coord
import numpy as np
from torchvision import transforms
from pdb import set_trace as bp
import time
import argparse


def init_meters(loss_str):
    losses = init_losses(loss_str)
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return losses, psnrs, ssims

def eval_metrics(output, gt, psnrs, ssims):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for b in range(gt.size(0)):
        psnr = calc_psnr(output[b], gt[b])
        psnrs.update(psnr)

        ssim = calc_ssim(output[b].unsqueeze(0).clamp(0,1), gt[b].unsqueeze(0).clamp(0,1) , val_range=1.)
        ssims.update(ssim)

def init_losses(loss_str):
    loss_specifics = {}
    loss_list = loss_str.split('+')
    for l in loss_list:
        _, loss_type = l.split('*')
        loss_specifics[loss_type] = AverageMeter()
    loss_specifics['total'] = AverageMeter()
    return loss_specifics

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)


def save_checkpoint(state, directory, is_best, exp_name, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory , filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory , 'model_best.pth'))

def log_tensorboard(writer, loss, psnr, ssim, lpips, lr, timestep, mode='train'):
    writer.add_scalar('Loss/%s/%s' % mode, loss, timestep)
    writer.add_scalar('PSNR/%s' % mode, psnr, timestep)
    writer.add_scalar('SSIM/%s' % mode, ssim, timestep)
    if mode == 'train':
        writer.add_scalar('lr', lr, timestep)
        
        
def test_images_2d(model, device, epoch=0, iter_id=0):
    model.eval()
    with torch.no_grad():
        out_path = '/output/Images/Epoch_{}'.format(epoch)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        # path_img1 = '/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/00000.jpg'
        # path_img2 = '/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/00001.jpg'
        path_img1 = '/data/nnice1216/vimeo_septuplet/sequences/00095/1000/im1.png'
        path_img2 = '/data/nnice1216/vimeo_septuplet/sequences/00095/1000/im2.png'

        img1 = Image.open(path_img1)
        img2 = Image.open(path_img2)

        h, w = img1.size
        #img1, img2 = img1.resize((h // 2, w // 2)), img2.resize((h // 2, w // 2))
        #h, w = h // 2, w // 2

        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)

        img1 = ((img1 - 0.5) * 2).to(device)
        img2 = ((img2 - 0.5) * 2).to(device) 
        h, w = img1.shape[1], img1.shape[2]
        coord = make_coord((h, w)).to(device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w

        #feat1 = model.module.gen_feat(img1[None])
        #feat2 = model.module.gen_feat(img2[None])    
        #feat = (feat1 + feat2) / 2
        #model.module.feat = feat2
    #     img = torch.stack([img1, img3, img2, img4], dim=1).view(-1, h, w)

    #     pred = model(img[None], coord[None], cell[None]).clamp(0, 1).view(1, h, w, 3).permute(0, 3, 1, 2)[0].cpu()
        img = torch.cat([img1, img2], dim=0)
        pred, _, _ = model(img[None], coord[None], cell[None])
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(1, h, w, 3).permute(0, 3, 1, 2)[0].cpu()
        pred = np.array(pred * 255).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(pred).save(os.path.join(out_path, 'iter{}.jpg'.format(iter_id)))
        
    return pred


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



def test_images_3d(model, device, time, epoch=0, iter_id=0):
    model.eval()
    with torch.no_grad():
        out_path = '/output/Images2/Epoch_{}'.format(epoch)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        imgpath = '/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/'
        imgpaths = [imgpath + f'/0000{i}.jpg' for i in range(1,8)]
        pth_ = imgpaths
        
        images = [Image.open(pth) for pth in imgpaths]
        h, w = images[0].size
        # images = [img.resize((360, 208)) for img in images]
        images = [img.resize((720, 416)) for img in images]
        inputs = [int(e)-1 for e in list('2345')]
        inputs = inputs[:len(inputs)//2] + inputs[len(inputs)//2:]
        images = [images[i] for i in inputs]
        imgpaths = [imgpaths[i] for i in inputs]
        
        T = transforms.ToTensor()
        # images = [((T(img_) - 0.5) * 2)[None] for img_ in images]
        images = [T(img_)[None] for img_ in images]
        h, w = images[0].shape[2], images[0].shape[3]
        coord = make_coord_3d((h, w), time)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell[:, 2] *= 0.5 
        coord, cell = coord.to(device), cell.to(device)
        
        images = [img_.to(device) for img_ in images]
                  
        torch.cuda.synchronize()
        
        out = model(images, coord[None], cell[None], False)

        torch.cuda.synchronize()
        out = (out).clamp(0, 1).view(1, h, w, 3).permute(0, 1, 2, 3)[0].cpu()
        Image.fromarray((out.numpy() * 255).astype(np.uint8)).save(os.path.join(out_path, 'Iter{}.jpg'.format(iter_id)))
        
        del out
        
        
def test_images_gopro(model, device, time, epoch=0, iter_id=0):
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        out_path = '/output/Images/Epoch_{}'.format(epoch)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        imgpath = '/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/'
        imgpaths = [imgpath + f'/0000{i}.jpg' for i in range(1,3)]
        pth_ = imgpaths
        
        images = [Image.open(pth) for pth in imgpaths]
        # images = [img.resize((360, 208)) for img in images]
        
        T = transforms.ToTensor()
        images = [T(img_)[None].to(device) for img_ in images]
                  
        torch.cuda.synchronize()
        
        out = model(images, [time])

        torch.cuda.synchronize()

        Image.fromarray((out[0][0].clamp(0., 1.).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)).save(os.path.join(out_path, 'Iter{}.jpg'.format(iter_id)))
        
        del out
        torch.cuda.empty_cache()
    
    
def test_images_outp(model, device, time, epoch=0, iter_id=0):
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        out_path = '/output/Image22sss/Epoch_{}'.format(epoch)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            
        imgpath = '/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/'
        imgpaths = [imgpath + f'/0000{i}.jpg' for i in range(1,8)]
        pth_ = imgpaths
        
        images = [Image.open(pth) for pth in imgpaths]
        h, w = images[0].size
        images = [img.resize((720, 416)) for img in images]
        inputs = [int(e)-1 for e in list('2345')]
        inputs = inputs[:len(inputs)//2] + inputs[len(inputs)//2:]
        images = [images[i] for i in inputs]
        imgpaths = [imgpaths[i] for i in inputs]
        
        T = transforms.ToTensor()
        # images = [((T(img_) - 0.5) * 2)[None] for img_ in images]
        images = [T(img_)[None] for img_ in images]
        h, w = images[0].shape[2], images[0].shape[3]
        coord = make_coord_3d((h, w), time)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell[:, 2] *= 0.5 
        coord, cell = coord.to(device), cell.to(device)
        
        images = [img_.to(device) for img_ in images]
        images = torch.stack(images, dim=2)
                  
        torch.cuda.synchronize()
        
        # out = model(images, coord[None], False) 
        out, loss_map, loss_map2 = model(images, coord[None], False)
        # loss_map = loss_map.view(1, h, w, 576).permute(0, 1, 2, 3)[0].sum(2)
        # loss_map = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min())
        torch.cuda.synchronize()
        Image.fromarray((loss_map[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_path, 'Iter{}_loss_map1.jpg'.format(iter_id)))
        Image.fromarray((loss_map2[0, 0].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_path, 'Iter{}_loss_map2.jpg'.format(iter_id)))
        Image.fromarray((out[0].clamp(0., 1.).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)).save(os.path.join(out_path, 'Iter{}.jpg'.format(iter_id)))
        
        del out
        torch.cuda.empty_cache()
        
        
def test_metric(model, epoch_id, iter_id, temp=False):
    torch.cuda.empty_cache()
    from dataset.GoPro import get_loader as gl
    test_loader = gl('train', '/data/nnice1216/high_FPS_video/GOPRO_Large_all/', 1, shuffle=False, num_workers=8, interFrames=7)    
    
    time_taken = []
    losses, psnrs, ssims = init_meters('1*L1')
    model.eval()

    psnr_list = []
    with torch.no_grad():
        start = time.time()
        for i, (images, gt_image, coords, cells, times) in enumerate(test_loader):
            # bp()
            if temp == True:
                if i > 100:
                    break
            if i % 10 == 0:
                print("TESTING, {} DONE!".format(i))
            
            images = [img_.cuda() for img_ in images]
            images = torch.stack(images, dim=2)
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
            if i == 15:
                out_dir = "/output/gopro_test/"
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                for j in range(7):
                    Image.fromarray((gt[j].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, 'Epoch{}_iter{}_{}_GT.jpg'.format(epoch_id, iter_id, j)))
                    Image.fromarray((out[j].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, 'Epoch{}_iter{}_{}_PRED.jpg'.format(epoch_id, iter_id, j)))
            '''
            torch.cuda.synchronize()
            time_taken.append(time.time() - start_time)

            eval_metrics(out, gt, psnrs, ssims)
        end = time.time()
    print('Epoch [{0}/{1}], Cost time: {2:.2f}s, Val_PSNR:{3:.4f}, Val_SSIM:{4:.4f}'
          .format(epoch_id, 30, end - start, psnrs.avg, ssims.avg))
    
    with open('/output/logs.txt', 'a') as f:
        print('Date: {0}s, Epoch: [{1}/{2}], Cost time: {3:.2f}s, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_id, iter_id, end - start, psnrs.avg, ssims.avg), file=f)
        
    # print("PSNR: %f, SSIM: %f" %
    #       (psnrs.avg, ssims.avg))
    # print("Time: " , sum(time_taken)/len(time_taken))
    torch.cuda.empty_cache()
    return psnrs.avg

def convert_to_gray(img):
    gray_img = img[:, 0] * 0.299 + img[:, 1] * 0.587 + img[:, 2] * 0.114
    return gray_img.unsqueeze(1)


def adjust_learning_rate(epoch):
    if epoch <= 2:
        lr = 1e-4
    elif epoch <= 5:
        lr = 5e-5
    elif epoch <= 15:
        lr = 2e-5
    else:
        lr = 1e-5
    
    return lr


def get_raft_args():
    parser_raft = argparse.ArgumentParser()
    parser_raft.add_argument('--model', default='/model/nnice1216/video/raft-small.pth', help="restore checkpoint")
    parser_raft.add_argument('--path', default='/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/', help="dataset for evaluation")
    parser_raft.add_argument('--small', default=True, help='use small model')
    parser_raft.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser_raft.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args1 = parser_raft.parse_known_args()[0]
    
    return args1
