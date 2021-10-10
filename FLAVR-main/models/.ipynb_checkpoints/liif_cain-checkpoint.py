import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

import models
from models import register
from utils import make_coord
import importlib
import argparse

from pdb import set_trace as bp
from models.FLAVR_arch import UNet_3D_3D
import numpy as np

from models.mlp import MLP
from models.SIREN import Siren
from models.cain import CAIN

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

from models.warplayer import warp, warpgrid

@register('liif_cain')
class LIIF_3d(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=False, feat_unfold=False, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        unet_3D = importlib.import_module(".resnet_3D" , "models")
        unet_3D.useBias = True
        self.encoder = CAIN()
        # self.encoder.load_state_dict(torch.load('/model/nnice1216/video/best.pth'))
        
        args1 = get_raft_args()
        self.raft = RAFT(args1)
        new_state_dict = OrderedDict()
        pre_trained_dict = torch.load(args1.model)
        for k, v in pre_trained_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.raft.load_state_dict(new_state_dict)
        self.raft.eval()
        # imnet_in_dim = 64
        # self.imnet_in_dim = imnet_in_dim
        # self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        
        imnet_in_dim2 = 257
        
        imnet_spec2 = {'name': 'mlp', 'args': {'out_dim': 27, 'hidden_list': [256, 64]}}
        # self.encode_imnet = models.make(imnet_spec2, args={'in_dim': imnet_in_dim2})
        self.encode_imnet = Siren(in_features=imnet_in_dim2, out_features=27, hidden_features=[64, 64, 256, 256, 256], hidden_layers=4, outermost_linear=True)
        # self.output_imnet = Siren(in_features=128, out_features=3, hidden_features=256, 
        #           hidden_layers=1, outermost_linear=True)
        
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
        # self.inp = torch.cat([inp[:, :, 0][None], inp[:, :, 1][None]], dim=0)
        self.inp = inp
        self.h = inp.shape[3]
        self.w = inp.shape[4]
        
        with torch.no_grad():
            padder = InputPadder(inp[:, :, 1].shape)
            image1, image2 = padder.pad(inp[:, :, 0].clamp(0, 1) * 255, inp[:, :, 1].clamp(0, 1) * 255)
            _, self.flow_f = self.raft(image1, image2, iters=20, test_mode=True)
            _, self.flow_b = self.raft(image2, image1, iters=20, test_mode=True)
        # self.mean_ = inp.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        # inp = inp - self.mean_ 
        # with torch.no_grad():p
        self.feat = self.encoder(inp[:, :, 0], inp[:, :, 1])
        # self.feat = torch.cat([self.encoder(inp), self.encoder(torch.flip(inp, dims=[2]))], dim=1)
        return self.feat

    def query_rgb(self, coord, Training=True):
        h, w = self.inp.shape[3], self.inp.shape[4]
        feat = self.feat
        # print(feat.shape)
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
        areas = []
        rets = []
        times = []
        ref_time_stamps = np.array([i / 8 for i in range(1, 8)])
        vx, vy = 0, 0
        if Training == True:
            prev_time = 0
            bs, qs = coord[0].shape[:2]
            coord_ = coord[0][:, :, :-1].clone()
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
            
            for c in range(3):
                
                time = coord[c][0, 0, -1].item()
                flow_t_to_0 = - time * (1 - time) * self.flow_f + (time ** 2) * self.flow_b
                grid, g2 = warpgrid(self.inp[:, :, 1], flow_t_to_0)
                grid = grid.view(grid.shape[0], -1, grid.shape[-1]).flip(-1)
                
                q_feat1 = F.grid_sample(
                        feat, grid.flip(-1).unsqueeze(1),
                        mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                        .permute(0, 2, 1)
                
                interval = coord[c][0, 0, -2].item()
                time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                fixed_time = ref_time_stamps[time_idxs]
                # pe_coord1 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * prev_time
                pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time 
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * fixed_time
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
                # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
               
                encode_inp = torch.cat([q_feat, q_feat1, pe_coord2], dim=-1)
                temp = self.encode_imnet(encode_inp.view(bs * qs, -1))
                # temporal_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1)
                # final_inp = torch.cat([q_feat, temporal_feat], dim=-1)
                pred = F.fold(temp.view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                
                # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                # pred = self.decode_conv(output_feat)
                # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
                # pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=1, padding=0)
                # pred = F.fold(self)
                # preds.append(pred + self.mean_.squeeze(2))
                preds.append(pred)
                # prev_time = time
                # q_feat = temporal_feat

            return preds
        
        elif Training == "testing":
            prev_time = 0
            bs, qs = coord[0].shape[:2]
            coord_ = coord[0][:, :, :-1].clone()
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
            
            for c in range(7):
                
                time = coord[c][0, 0, -1].item()
                flow_t_to_0 = - time * (1 - time) * self.flow_f + (time ** 2) * self.flow_b
                grid, g2 = warpgrid(self.inp[:, :, 1], flow_t_to_0)
                grid = grid.view(grid.shape[0], -1, grid.shape[-1]).flip(-1)
                
                interval = coord[c][0, 0, -2].item()
                time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                fixed_time = ref_time_stamps[time_idxs]
                
                q_feat1 = F.grid_sample(
                        feat, grid.flip(-1).unsqueeze(1),
                        mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                        .permute(0, 2, 1)
                
                # pe_coord1 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * prev_time
                pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time 
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * fixed_time
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
                # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                # bp()
                encode_inp = torch.cat([q_feat, pq_feat1, e_coord2], dim=-1)
                # temporal_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1)
                # final_inp = torch.cat([q_feat, temporal_feat], dim=-1)
                pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                
                # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                # pred = self.decode_conv(output_feat)
                # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
                # pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=1, padding=0)
                # preds.append(pred + self.mean_.squeeze(2))
                preds.append(pred)
                # prev_time = time
                # q_feat = temporal_feat

            return preds
    
        elif Training == 8:
            prev_time = 0
            bs, qs = coord[0].shape[:2]
            coord_ = coord[0][:, :, :-1].clone()
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
            
            for c in range(8):
                for i in range(8):
                    time = c / 8 + i / 64
                    if time == 0.0:
                        continue
                    time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                    fixed_time = ref_time_stamps[time_idxs]
                    # pe_coord1 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * prev_time
                    pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time 
                    # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * fixed_time
                    # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
                    # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                    # bp()
                    encode_inp = torch.cat([q_feat, pe_coord2], dim=-1)
                    # temporal_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1)
                    # final_inp = torch.cat([q_feat, temporal_feat], dim=-1)
                    pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)

                    # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                    # pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=1, padding=0)
                    # preds.append(pred + self.mean_.squeeze(2))
                    preds.append(pred)
                    # prev_time = time
                    # q_feat = temporal_feat

            return preds

        else:
            prev_time = 0
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
            flow_t_to_0 = - time * (1 - time) * self.flow_f + (time ** 2) * self.flow_b
            grid, g2 = warpgrid(self.inp[:, :, 1], flow_t_to_0)
            grid = grid.view(grid.shape[0], -1, grid.shape[-1]).flip(-1)
            time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
            fixed_time = ref_time_stamps[time_idxs]
            q_feat1 = F.grid_sample(
                        feat, grid.flip(-1).unsqueeze(1),
                        mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                        .permute(0, 2, 1)
            
            pe_coord1 = torch.ones_like(coord[:, :, -1].unsqueeze(2)) * prev_time
            pe_coord2 = torch.ones_like(coord[:, :, -1].unsqueeze(2)) * time 
            # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * fixed_time
            # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
            # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
            # bp()
            encode_inp = torch.cat([q_feat, q_feat1, pe_coord2], dim=-1)
            # temporal_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1)
            # final_inp = torch.cat([q_feat, temporal_feat], dim=-1)
            pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                
            # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
            # pred = self.decode_conv(output_feat)
            # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
            # pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=1, padding=0)
            
            # return pred + self.mean_.squeeze(2)
            return pred
        

    def forward(self, inp, coord, Training=True):
        self.gen_feat(inp)
        return self.query_rgb(coord, Training)
    
    
def get_raft_args():
    parser_raft = argparse.ArgumentParser()
    parser_raft.add_argument('--model', default='/model/nnice1216/video/raft-small.pth', help="restore checkpoint")
    parser_raft.add_argument('--path', default='/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/', help="dataset for evaluation")
    parser_raft.add_argument('--small', default=True, help='use small model')
    parser_raft.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser_raft.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args1 = parser_raft.parse_known_args()[0]
    
    return args1