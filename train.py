import os
import cv2
import copy
import random
import numpy as np

from utils import *
from data_utils import *
from models.network_dncnn import DnCNN
from PDNet import DnCNN_c, Estimation_direct, DecomNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def post_process(denoised_img, original_img):
    color = 1
    pss = 2
    rescale = 1
    w = 400
    h = 600
    c = 3
    mosaic_den = visual_va2np(denoised_img, color, 1, pss, 1, rescale, w, h, c)
    out_numpy = np.zeros((pss ** 2, c, w, h))
    output_path = '/output/temp_imgs/' 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with torch.no_grad():
        for row in range(pss):
            for column in range(pss):
                re_test = visual_va2np(denoised_img, color, 1, pss, 1, rescale, w, h, c, 1, visual_va2np(original_img, color), [row, column])/255.
                #cv2.imwrite(output_path + '_%d_%d.png' % (row, column), re_test[:,:,::-1]*255.)
                re_test = np.expand_dims(re_test, 0)

                re_test_tensor = torch.from_numpy(np.transpose(re_test, (0,3,1,2))).type(torch.FloatTensor)
                re_test_tensor = Variable(re_test_tensor.cuda(),volatile=True)
                est = torch.clamp(est_net(re_test_tensor), 0., 1.)    
                re_Res = model(re_test_tensor, est)
                Out2 = torch.clamp(re_test_tensor - re_Res, 0., 1.)
                Out2 = Out2.data.cpu().numpy()

                cv2.imwrite(output_path + '_%d_%d.png' % (row, column), Out2[0].transpose(1, 2, 0)[:,:,::-1]*255.)
                out_numpy[row*pss+column,:,:,:] = Out2                                                         
                del Out2
                del re_Res
                del re_test_tensor
                del re_test

        out_numpy = np.mean(out_numpy, 0)
        cv2.imwrite(output_path + 'mean.png', out_numpy.transpose(1, 2, 0)[:,:,::-1]*255.)
    return out_numpy

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DnCNN_c(channels=3, num_of_layers=20, num_of_est = 2 * c)
model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)
model.load_state_dict(torch.load('./original_model/dncnn_net.pth'))

est_net = Estimation_direct(c, 2 * c)
est_net.to(device)
est_net = nn.DataParallel(est_net, device_ids=device_ids)
est_net.load_state_dict(torch.load('./original_model/est_net.pth'))
net_o = copy.deepcopy(est_net)
net_o.eval()

decom_net = DecomNet()
decom_net.to('cuda')
decom_net = nn.DataParallel(decom_net, device_ids=device_ids)
decom_net.load_state_dict(torch.load('./original_model/decom_net.pth'))
decom_net.eval()

crop_size = 60
batch_size = 16
num_epochs = 6

data_dir = './data/semi_train/'
train_low_data_dir = './data/semi_train/low/'
train_loader = FinetuneData_with_augment_newdata(crop_size, data_dir, train_low_data_dir, 10, 0, 0, False)
train_data_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    
lr = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
print("BEGIN")
epoch = 0
for epoch in range(num_epochs):
    for batch_id, data in enumerate(train_data_loader):
        _, noisy_img, gt_img, low_img = data
        
        noisy_img = noisy_img.to(device)
        gt_img = gt_img.to(device)
        low_img = low_img.to(device)
        
        _, R_low, I_low = decom_net(low_img)
        _, R_en, I_en = decom_net(noisy_img)
        
        est = torch.clamp(net_o(noisy_img), 0., 1.).clone().detach()
        noise_level = (est[0, :3].mean() * 100).item()
        noise_map = torch.normal(0, noise_level /255., noisy_img.shape).type(torch.FloatTensor).to(device)
        est_3 = torch.cat([est[:, :3].mean(1)[:, None], est[:, :3].mean(1)[:, None], est[:, :3].mean(1)[:, None]], dim=1)
        noise_map = noise_map * (1 + est_3)
        noisy_noisy_img = torch.clamp(noisy_img + noise_map, 0., 1.)
        
        noise_level = 10
        optimizer.zero_grad()
        
        res = model(noisy_noisy_img, est)
        denoised_img1 = torch.clamp(noisy_noisy_img - res, 0., 1.)
        est2 = torch.clamp(est_net(noisy_img), 0., 1.)
        res2 = model(noisy_img, est2)
        res3 = model(noisy_noisy_img, est2)
        denoised_img2 = torch.clamp(noisy_img - res3, 0., 1.)
        denoised_img3 = torch.clamp(noisy_img - res2, 0., 1.)
        _, R_de, I_de = decom_net(denoised_img3)
        I_en_3 = torch.cat((I_en, I_en, I_en), 1)
        
        loss1 = F.mse_loss(denoised_img1, noisy_img)
        loss2 = F.mse_loss(R_low * I_en_3, denoised_img2) + F.mse_loss(R_low * I_en_3, denoised_img3)
        
        if loss1.item() > 10:
            print(loss1.item())
            continue
            
        loss = loss1 + 0.3 * loss2
        print('\r', "Epoch: %d, Iter: %d, loss: %f, loss1: %f, loss2: %f"%\
              (epoch, batch_id, loss.item(), loss1.item(), 0.3 * loss2.item()), end='')
        loss.backward()
        optimizer.step()
    print('\n')

model_dir = './checkpoints'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
torch.save(model.state_dict(), os.path.join(model_dir, 'dncnn_net.pth'))
torch.save(est_net.state_dict(), os.path.join(model_dir, 'est_net.pth'))
torch.save(decom_net.state_dict(), os.path.join(model_dir, 'decom_net.pth'))
