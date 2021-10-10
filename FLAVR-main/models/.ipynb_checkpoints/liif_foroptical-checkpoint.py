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


@register('liif_foroptical')
class LIIF_3d(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=False, feat_unfold=False, cell_decode=False):
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
        
        imnet_in_dim2 = 65
        
        imnet_spec2 = {'name': 'mlp', 'args': {'out_dim': 27, 'hidden_list': [256, 64]}}
        # self.encode_imnet = models.make(imnet_spec2, args={'in_dim': imnet_in_dim2})
        self.encode_imnet = Siren(in_features=imnet_in_dim2, out_features=18, hidden_features=[64, 64, 256, 256, 256], 
                  hidden_layers=4, outermost_linear=True)
        
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
    
    def gen_feat(self, inp, Training):

        # inp = torch.stack(inp, dim=2)
        self.h = inp.shape[3]
        self.w = inp.shape[4]

        self.feat = self.encoder(inp)
        
        return 

    def query_rgb(self, coord, Training=True):
        feat = self.feat
        # feat = feat.view(feat.shape[0], 64, 9, feat.shape[2], feat.shape[3])
        # feat = self.outconv(self.feat)
        # feat = self.outconv(feat).view(feat.shape[0], 64, 9, feat.shape[2], feat.shape[3])
        feats = []
        
        # feat = torch.cat(feats, dim=0).permute(1, 2, 0, 3, 4)
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        
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

        preds = []
        preds2 = []
        areas = []
        rets = []
        times = []
        ref_time_stamps = np.array([i / 8 for i in range(1, 8)])
        vx, vy = 0, 0
        
        if Training == True:
            for c in range(1):
                bs, qs = coord[c].shape[:2]
                coord_ = coord[c][:, :, :-1].clone()
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
                rel_coord = coord[c][:, :, :-2] - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                
                time = coord[c][0, 0, -1].item()
                interval = coord[c][0, 0, -2].item()
                time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                pe_coord = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time
                pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * (time - 1)
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
                # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                
                encode_inp = torch.cat([q_feat, pe_coord], dim=-1)
                temp = self.encode_imnet(encode_inp.view(bs * qs, -1))
                # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                # pred = self.decode_conv(output_feat)
                # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
                pred = F.fold(temp.view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                preds.append(pred)
                
            return preds
        
        elif Training == "testing":
            for c in range(8):
                
                bs, qs = coord[c].shape[:2]
                coord_ = coord[c][:, :, :-1].clone()
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
                rel_coord = coord[c][:, :, :-2] - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                
                time = coord[c][0, 0, -1].item()
                interval = coord[c][0, 0, -2].item()
                time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                pe_coord = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
                # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                
                encode_inp = torch.cat([q_feat, pe_coord], dim=-1)
                
                # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                # pred = self.decode_conv(output_feat)
                # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
                pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2] * 4, feat.shape[3] * 4), kernel_size=3, padding=1)
                # preds.append(pred + self.mean_.squeeze(2))
                preds.append(pred)

            return preds
    
        elif Training == 8:
            for c in range(8):
                for i in range(8):
                    time = c / 8 + i / 64
                    if time == 0.0:
                        continue
                
                    bs, qs = coord[c].shape[:2]
                    coord_ = coord[c][:, :, :-1].clone()
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
                
                    time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                    pe_coord = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time
                    pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * (1 / 64)
                    # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()

                    encode_inp = torch.cat([q_feat, pe_coord], dim=-1)

                    # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                    pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2] * 2, feat.shape[3] * 2), kernel_size=3, padding=1)
                    # preds.append(pred + self.mean_.squeeze(2))
                    preds.append(pred)

            return preds

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
            rel_coord = coord[:, :, :-1] - q_coord
            rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]
                
            time = coord[0, 0, -1].item()
            time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
            pe_coord = torch.ones_like(coord[:, :, -1].unsqueeze(2)) * time
            pe_coord2 = torch.ones_like(coord[:, :, -1].unsqueeze(2)) * (1 / 100)
            #pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                
            encode_inp = torch.cat([q_feat, pe_coord], dim=-1)
                
            # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
            # pred = self.decode_conv(output_feat)
            # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
            pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
            # return pred + self.mean_.squeeze(2)
            return pred
        

    def forward(self, inp, coord, Training=True):
        self.gen_feat(inp, Training)
        return self.query_rgb(coord, Training)