import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

import models
from models import register
from utils import make_coord
import importlib

from pdb import set_trace as bp
from models.FLAVR_arch import UNet_3D_3D
import numpy as np

from models.mlp import MLP
from models.SIREN import Siren

from PIL import Image
import os
import myutils


@register('liif_bidi')
class LIIF_3d(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=False, feat_unfold=True, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        unet_3D = importlib.import_module(".resnet_3D" , "models")
        unet_3D.useBias = True
        self.encoder = UNet_3D_3D('unet_18', n_inputs=4, n_outputs=7, joinType='concat')
        new_state_dict = OrderedDict()
        pre_trained_dict = torch.load('/model/nnice1216/video/FLAVR_8x.pth')['state_dict']
        for k, v in pre_trained_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        
        self.encoder.load_state_dict(new_state_dict)
        # self.encoder.load_state_dict(torch.load('/model/nnice1216/video/FLAVR_2x.pth')['state_dict'])
        # model_dict = self.encoder.state_dict()
        # pre_trained_dict = torch.load('/model/nnice1216/video/FLAVR_8x.pth')['state_dict']
        # pre_trained_dict = {k[7:]: v for k, v in pre_trained_dict.items() if k[7:] in model_dict}
        # print(pre_trained_dict.keys())
        # model_dict.update(pre_trained_dict)
        # self.encoder.load_state_dict(model_dict)
    
        # imnet_in_dim = 64
        # self.imnet_in_dim = imnet_in_dim
        # self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        
        imnet_in_dim2 = 577
        
        imnet_spec2 = {'name': 'mlp', 'args': {'out_dim': 27, 'hidden_list': [256, 64]}}
        # self.encode_imnet = models.make(imnet_spec2, args={'in_dim': imnet_in_dim2})
        self.encode_imnet = Siren(in_features=imnet_in_dim2, out_features=27, hidden_features=[256, 256, 256, 256], 
                  hidden_layers=3, outermost_linear=True)
#         self.out_conv = nn.Sequential(
#                 nn.ReflectionPad2d(3),
#                 nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0)
#         )
        # self.encode_imnet = models.make(imnet_spec2, args={'in_dim': imnet_in_dim2})
        '''
        self.decode_conv = nn.Sequential(
                # nn.Conv2d(256, 64 , kernel_size=3 , stride=1, padding=1),
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0)
            )
        
        
        self.outconvs = []
        for i in range(13):
            self.outconvs.append(nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 64 , kernel_size=7 , stride=1, padding=0) 
            ))
        self.outconvs = nn.ModuleList(self.outconvs)
        
        self.tmb = TMB()
        
        self.outconv = nn.Sequential(
                nn.ReflectionPad2d(2),
                nn.Conv2d(64, 576, kernel_size=5 , stride=1, padding=2) 
            )
        '''
        
        
        
    def position_encoding(self, p, L=12):
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(1)
        w = torch.exp(position)
        pe = torch.zeros(2 * L, 1)
        pe[0::2] = torch.sin(w * p)
        pe[1::2] = torch.cos(w * p)

        return pe    
    
    def gen_feat(self, inp):

        # inp = torch.stack(inp, dim=2)
        self.h = inp.shape[3]
        self.w = inp.shape[4]
        
        # self.mean_ = inp[:, :, 1:3].mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        # inp = inp - self.mean_ 
        # self.mean_ = self.mean_.squeeze(2)
        # with torch.no_grad():
        self.feat_forward = self.encoder(inp)
        self.feat_backward = self.encoder(torch.flip(inp, dims=[2]))
        
        return
    
    def query_rgb(self, coord, Training=True, index=0):
        feat = self.feat_forward
        feat_backward = self.feat_backward
        # feat = feat.view(feat.shape[0], 64, 9, feat.shape[2], feat.shape[3])
        # feat = self.outconv(self.feat)
        # feat = self.outconv(feat).view(feat.shape[0], 64, 9, feat.shape[2], feat.shape[3])
        feats = []
        
        # feat = torch.cat(feats, dim=0).permute(1, 2, 0, 3, 4)
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            feat_backward = F.unfold(feat_backward, 3, padding=1).view(
                feat_backward.shape[0], feat_backward.shape[1] * 9, feat_backward.shape[2], feat_backward.shape[3])
            
        # del feat_backward, self.feat_backward
        # feat = torch.cat([feat[0], feat[1], feat[2], feat[3]], dim=1)
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds_forward = []
        preds_backward = []
        preds = []
        areas = []
        rets = []
        times = []
        ref_time_stamps = np.array([i / 8 for i in range(1, 8)])
        vx, vy = 0, 0
        
        if Training == True:
            bs, qs = coord[0].shape[:2]
            coord_ = coord[0][:, :, :-1].clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
            q_feat_forward = F.grid_sample(
                feat, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat_backward = F.grid_sample(
                feat_backward, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            
            for c in range(len(coord)):
                time = coord[c][0, 0, -1].item()
                interval = coord[c][0, 0, -2].item()
                time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                pe_coord = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time
                pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * (1 - time)
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
                # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                
                encode_inp_forward = torch.cat([q_feat_forward, pe_coord], dim=-1)
                encode_inp_backward = torch.cat([q_feat_backward, pe_coord2], dim=-1)
                pred_forward = self.out_conv(self.encode_imnet(encode_inp_forward.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3]))
                pred_backward = self.out_conv(self.encode_imnet(encode_inp_backward.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3]))
                # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                # pred = self.decode_conv(output_feat)
                # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
                # pred_forward = F.fold(self.encode_imnet(encode_inp_forward.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                # pred_backward = F.fold(self.encode_imnet(encode_inp_backward.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                # pred = F.fold(self)
                # preds.append(pred + self.mean_.squeeze(2))
                preds_forward.append(pred_forward)
                preds_backward.append(pred_backward)
            
            torch.cuda.empty_cache()
            return preds_forward, preds_backward
        
        elif Training == "testing":
            for c in range(7):
                
                bs, qs = coord[c].shape[:2]
                coord_ = coord[c][:, :, :-1].clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                
                time = coord[c][0, 0, -1].item()
                interval = coord[c][0, 0, -2].item()
                time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                pe_coord = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
                # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                
                encode_inp = torch.cat([q_feat, pe_coord], dim=-1)
                pred = F.fold(self.encode_imnet(pred.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                # pred = self.decode_conv(output_feat)
                # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
                # pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                # preds.append(pred + self.mean_.squeeze(2))
                preds.append(pred + self.mean_)

            return preds
    
        elif Training == 8:
            for c in range(8):
                for i in range(8):
                    time = c / 8 + i / 64
                    if time == 0.0:
                        continue
                
                    bs, qs = coord[0].shape[:2]
                    coord_ = coord[0][:, :, :-1].clone()
                    coord_[:, :, 0] += vx * rx + eps_shift
                    coord_[:, :, 1] += vy * ry + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                
                    time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                    pe_coord = torch.ones_like(coord[0][:, :, -1].unsqueeze(2)) * time
                    # pe_coord2 = torch.ones_like(coord[0][:, :, -1].unsqueeze(2)) * (1 / 64)
                    # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()

                    pred = torch.cat([q_feat, pe_coord], dim=-1)
                    pred = F.fold(self.encode_imnet(pred.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                    del q_feat, pe_coord
                    # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                    # pred = F.fold(self.encode_imnet(pred.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                    if i == 7 and c == 7:
                        return pred 
                    else:
                        pred = (pred).clamp(0, 1).permute(0, 2, 3, 1)[0].cpu()
                        save_image(Image.fromarray((pred.numpy() * 255).astype(np.uint8)), index)
                        print("INDEX {}, DONE!".format(index))
                        index += 1
                    torch.cuda.empty_cache() 
                    # preds.append(pred + self.mean_.squeeze(2))
                    

            return torch.tensor(index)

        else:
            
            bs, qs = coord.shape[:2]
            coord_ = coord[:, :, :-1].clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
            q_feat = F.grid_sample(
                feat, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_coord = F.grid_sample(
                feat_coord, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
                
            time = coord[0, 0, -1].item()
            time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
            pe_coord = torch.ones_like(coord[:, :, -1].unsqueeze(2)) * time
            #pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                
            encode_inp = torch.cat([q_feat, pe_coord], dim=-1)
            pred = F.fold(self.encode_imnet(pred.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                
            # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
            # pred = self.decode_conv(output_feat)
            # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
            # pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
            
            # return pred + self.mean_.squeeze(2)
            torch.cuda.empty_cache()
            return pred + self.mean_
        

    def forward(self, inp, coord, Training=True, index=0):
        self.gen_feat(inp)
        return self.query_rgb(coord, Training, index)
    

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
    out_path1 = '/output/Imagestry11/'
    out_path2 = '/output/Imagestry22/'
    out_path3 = '/output/Imagestry33/'
    out_path4 = '/output/Imagestry44/'
    out_path5 = '/output/Imagestry55/'
    out_path6 = '/output/Imagestry66/'
    out_path7 = '/output/Imagestry77/'
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