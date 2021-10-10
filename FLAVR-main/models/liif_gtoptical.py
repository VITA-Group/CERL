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

from PIL import Image
import os
import myutils

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
from pdb import set_trace as bp

from models.warplayer import warp, warpgrid

from gpu_memory_log import gpu_memory_log



@register('liif_gtoptical')
class LIIF_3d(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=False, feat_unfold=True, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.device = 'cuda'
        self.map_size = 19
        self.map_size2 = 19
        
        unet_3D = importlib.import_module(".resnet_3D" , "models")
        unet_3D.useBias = True
        self.encoder = UNet_3D_3D('unet_18', n_inputs=4, n_outputs=7, joinType='concat')
        new_state_dict = OrderedDict()
        pre_trained_dict = torch.load('/model/nnice1216/video/FLAVR_8x.pth')['state_dict']
        for k, v in pre_trained_dict.items():
            name = k[7:]
            new_state_dict[name] = v
            
        self.encoder.load_state_dict(new_state_dict)

        args1 = get_raft_args()
        self.raft = RAFT(args1)
        new_state_dict = OrderedDict()
        pre_trained_dict = torch.load(args1.model, map_location=torch.device('cpu'))
        for k, v in pre_trained_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.raft.load_state_dict(new_state_dict)
        self.raft.eval()
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
        
        imnet_in_dim = 579
        imnet_in_dim2 = 577
        ode_dim = 64
        base_dim = 256
        
        imnet_spec2 = {'name': 'mlp', 'args': {'out_dim': 27, 'hidden_list': [256, 64]}}
        # self.encode_imnet = models.make(imnet_spec2, args={'in_dim': imnet_in_dim2})
        self.mask_imnet = Siren(in_features=imnet_in_dim, out_features=1, hidden_features=[64, 64, 256, 256], 
                  hidden_layers=3, outermost_linear=True)
        self.encode_imnet = Siren(in_features=imnet_in_dim2, out_features=27, hidden_features=[64, 64, 256, 256, 256], 
                  hidden_layers=4, outermost_linear=True)
        

        
        
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
        with torch.no_grad():
            padder = InputPadder(inp[:, :, 1].shape)
            image1, image2 = padder.pad(inp[:, :, 1].clamp(0, 1) * 255, inp[:, :, 2].clamp(0, 1) * 255)
            _, self.flow_f = self.raft(image1, image2, iters=20, test_mode=True)
            _, self.flow_b = self.raft(image2, image1, iters=20, test_mode=True)
                
        # self.flow_f = (self.flow_f - self.flow_f.min()) / (self.flow_f.max() - self.flow_f.min())
        # self.flow_b = (self.flow_b - self.flow_b.min()) / (self.flow_b.max() - self.flow_b.min())
        # self.flow_f[:, 0], self.flow_b[:, 0] = self.flow_f[:, 0] / self.h, self.flow_b[:, 0] / self.h
        # self.flow_f[:, 1], self.flow_b[:, 1] = self.flow_f[:, 1] / self.w, self.flow_b[:, 1] / self.w
        # with torch.no_grad():
        # self.feat_forward = torch.cat([self.encoder(inp), flow_f, flow_b], dim=1)
        # self.feat_backward = torch.cat([self.encoder(torch.flip(inp, dims=[2])), flow_b, flow_f], dim=1)
        self.feat = self.encoder(inp)
        # self.feat = torch.cat([self.encoder(inp), self.encoder(torch.flip(inp, dims=[2]))], dim=1)
        # self.feat_backward = self.encoder(torch.flip(inp, dims=[2]))
        
        
        return
    
    def query_rgb(self, coord, Training=True, index=0):
        feat = self.feat
        feats = []
        
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            
        del self.feat
        torch.cuda.empty_cache() 
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
        feats = []
        warps = []
        masks = []
        flow_dists1 = []
        flow_dists2 = []
        areas = []
        rets = []
        times = []
        flows = []
        ref_time_stamps = np.array([i / 8 for i in range(1, 8)])
        vx, vy = 0, 0
        if Training == True:
            prev_time = 0
            bs, qs = coord[0].shape[:2]
            coord_ = coord[0][:, :, :-1].clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
            
            for c in range(3):
                
                time = coord[c][0, 0, -1].item()
                with torch.no_grad():
                    padder = InputPadder(self.inp[:, :, 1].shape)
                    image1, image2 = padder.pad(self.gts[c].detach().clamp(0, 1) * 255, self.inp[:, :, 1].detach().clamp(0, 1) * 255)
                    _, flow_t_to_0_gt = self.raft(image1, image2, iters=20, test_mode=True)
                    
                flow_t_to_0 = - time * (1 - time) * self.flow_f + (time ** 2) * self.flow_b
                # flow_t_to_1 = ((1 - time) ** 2) * self.flow_f - time * (1 - time) * self.flow_b
                
                flow_map_gt, _ = self.generate_flow_grad(flow_t_to_0_gt, coord_)
                flow_map1, q_flow1 = self.generate_flow_grad(flow_t_to_0, coord_)
                # flow_map2, q_flow2 = self.generate_flow_grad(flow_t_to_1, coord_)
                
                grid1, _ = warpgrid(self.inp[:, :, 1], flow_t_to_0)
                # grid2, _ = warpgrid(self.inp[:, :, 2], flow_t_to_1)
                grid = grid1.view(grid1.shape[0], -1, grid1.shape[-1]).flip(-1)
                # grid2 = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1)
                del flow_t_to_0
                torch.cuda.empty_cache()
                
                q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                        .permute(0, 2, 1)
                q_feat1 = F.grid_sample(
                        feat, grid.flip(-1).unsqueeze(1),
                        mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                        .permute(0, 2, 1)
#                 q_feat2 = F.grid_sample(
#                         feat, grid2.flip(-1).unsqueeze(1),
#                         mode='nearest*', padding_mode="border", align_corners=True)[:, :, 0, :] \
#                         .permute(0, 2, 1)
                # warped_img = F.grid_sample(input=self.inp[:, :, 1], grid=grid1, mode='bilinear', padding_mode='border', align_corners=True)
                # warped_img = convert_to_gray(warped_img)
                interval = coord[c][0, 0, -2].item()
                time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                fixed_time = ref_time_stamps[time_idxs]
                # pe_coord1 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * prev_time
                pe_coord = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * fixed_time
                # bp()
                mask_inp = torch.cat([q_feat, grid.flip(-1), pe_coord], dim=-1)
                mask_map = F.sigmoid(F.fold(self.mask_imnet(mask_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=1, padding=0) * 8)
                ones_map = torch.ones_like(mask_map)
                zeros_map = torch.zeros_like(mask_map)
                mask_map = torch.where(mask_map > 0, ones_map, zeros_map)
                mask = mask_map.view(bs, qs, -1)
                encode_inp = torch.cat([q_feat * mask + (1 - mask) * q_feat1, pe_coord], dim=-1)
                # temporal_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1)
                # final_inp = torch.cat([q_feat, temporal_feat], dim=-1)
                # pred = F.fold(self.output_imnet(final_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=1, padding=0)
                
                # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                # pred = self.decode_conv(output_feat)
                # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
                pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                # pred = F.fold(self)
                # preds.append(pred + self.mean_.squeeze(2))
                preds.append(pred)
                masks.append(mask_map)
                flows.append(flow_map_gt.unsqueeze(1))
                # feats.append([q_feat, q_feat1])
                
                # flow_dists1.append(flow_map1.detach())
                # flow_dists2.append(flow_map2.detach())
                
                # flow_dist = torch.sqrt(flow_t_to_0[:, 0] ** 2 + flow_t_to_0[:, 1] ** 2)
                # norm_flow_dist = (flow_dist - flow_dist.min()) / (flow_dist.max() - flow_dist.min())
                # flow_dists.append(norm_flow_dist.detach())
                prev_time = time
                # q_feat = temporal_feat

            return preds, masks, flows
        
        elif Training == "testing":
            prev_time = 0
            bs, qs = coord[0].shape[:2]
            coord_ = coord[0][:, :, :-1].clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
            
            for c in range(7):
                
                time = coord[c][0, 0, -1].item()
                # flow_t_to_0 = time * self.flow_f
                flow_t_to_0 = - time * (1 - time) * self.flow_f + (time ** 2) * self.flow_b
                flow_t_to_1 = ((1 - time) ** 2) * self.flow_f - time * (1 - time) * self.flow_b
                _, q_flow1 = self.generate_flow_grad(flow_t_to_0, coord_)
                _, q_flow2 = self.generate_flow_grad(flow_t_to_1, coord_)
                
                grid, _ = warpgrid(self.inp[:, :, 1], flow_t_to_0)
                grid2, _ = warpgrid(self.inp[:, :, 2], flow_t_to_1)
                grid = grid.view(grid.shape[0], -1, grid.shape[-1]).flip(-1)
                grid2 = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1)
                
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat1 = F.grid_sample(
                    feat, grid.flip(-1).unsqueeze(1),
                    mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat2 = F.grid_sample(
                    feat, grid2.flip(-1).unsqueeze(1),
                    mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, grid.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=True)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                interval = coord[c][0, 0, -2].item()
                time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                fixed_time = ref_time_stamps[time_idxs]
                # pe_coord1 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * prev_time
                pe_coord = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * fixed_time
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
                # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                # bp()
                encode_inp = torch.cat([q_flow1 * q_feat + (1 - q_flow1) * q_feat1, q_flow2 * q_feat + (1 - q_flow2) * q_feat2, pe_coord], dim=-1)
                # temporal_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1)
                # final_inp = torch.cat([q_feat, temporal_feat], dim=-1)
                # pred = F.fold(self.output_imnet(final_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=1, padding=0)
                
                # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                # pred = self.decode_conv(output_feat)
                # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
                pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                # preds.append(pred + self.mean_.squeeze(2))
                preds.append(pred)
                prev_time = time
                # q_feat = temporal_feat

            return preds
        
        elif Training == 8:
            prev_time = 0
            bs, qs = coord[0].shape[:2]
            coord_ = coord[0][:, :, :-1].clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
            
            for c in range(8):
                for i in range(8):
                    time = c / 8 + i / 64
                    if time == 0.0:
                        continue
                        
                    flow_t_to_0 = - time * (1 - time) * self.flow_f + (time ** 2) * self.flow_b
                    # flow_t_to_1 = ((1 - time) ** 2) * self.flow_f - time * (1 - time) * self.flow_b
                    _, q_flow1 = self.generate_flow_grad(flow_t_to_0, coord_)
                    del _
                    torch.cuda.empty_cache() 
                    # _, q_flow2 = self.generate_flow_grad(flow_t_to_1, coord_)
                    # del _
                    # torch.cuda.empty_cache() 
                    
                    grid, _ = warpgrid(self.inp[:, :, 1], flow_t_to_0)
                    # grid2, _ = warpgrid(self.inp[:, :, 2], flow_t_to_1)
                    del flow_t_to_0
                    torch.cuda.empty_cache() 
                    grid = grid.view(grid.shape[0], -1, grid.shape[-1]).flip(-1)
                    # grid2 = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1)
                    q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                        .permute(0, 2, 1)
                    q_feat1 = F.grid_sample(
                        feat, grid.flip(-1).unsqueeze(1),
                        mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                        .permute(0, 2, 1)
#                     q_feat2 = F.grid_sample(
#                         feat, grid2.flip(-1).unsqueeze(1),
#                         mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
#                         .permute(0, 2, 1)
                    
                    time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                    fixed_time = ref_time_stamps[time_idxs]
                    # pe_coord1 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * prev_time
                    pe_coord = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * time
                    # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * fixed_time
                    # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
                    # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                    # bp()
                    encode_inp = torch.cat([q_feat, q_feat1 * (1 - q_flow1), pe_coord], dim=-1)
                    del q_feat, q_feat1
                    torch.cuda.empty_cache() 
                    # temporal_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1)
                    # final_inp = torch.cat([q_feat, temporal_feat], dim=-1)
                    # pred = F.fold(self.output_imnet(final_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=1, padding=0)

                    # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                    pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                    # preds.append(pred + self.mean_.squeeze(2))
                    if i == 7 and c == 7:
                        return pred
                    else:
                        pred = pred.clamp(0, 1).permute(0, 2, 3, 1)[0].cpu()
                        save_image(Image.fromarray((pred.numpy() * 255).astype(np.uint8)), index)
                        print("INDEX {}, DONE!".format(index))
                        index += 1
                        
                    del pred
                    torch.cuda.empty_cache() 
                    prev_time = time
                    # q_feat = temporal_feat
            
            return preds

        else:
            prev_time = 0
            bs, qs = coord.shape[:2]
            coord_ = coord[:, :, :-1].clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
            time = coord[0, 0, -1].item()
            # flow_t_to_0 = time * self.flow_f
            flow_t_to_0 = - time * (1 - time) * self.flow_f + (time ** 2) * self.flow_b
            # flow_t_to_1 = ((1 - time) ** 2) * self.flow_f - time * (1 - time) * self.flow_b
            flow_map1, q_flow1 = self.generate_flow_grad(flow_t_to_0, coord_, True)
            # flow_map2, q_flow2 = self.generate_flow_grad(flow_t_to_1, coord_, True)
                
            grid1, _ = warpgrid(self.inp[:, :, 1], flow_t_to_0)
            # grid2, _ = warpgrid(self.inp[:, :, 2], flow_t_to_1)
            grid = grid1.view(grid1.shape[0], -1, grid1.shape[-1]).flip(-1)
            # grid2 = grid2.view(grid2.shape[0], -1, grid2.shape[-1]).flip(-1)
            
            q_feat = F.grid_sample(
                feat, coord_.flip(-1).unsqueeze(1),
                mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                .permute(0, 2, 1)
            q_feat1 = F.grid_sample(
                feat, grid.flip(-1).unsqueeze(1),
                mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                .permute(0, 2, 1)
#             q_feat2 = F.grid_sample(
#                         feat, grid2.flip(-1).unsqueeze(1),
#                         mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
#                         .permute(0, 2, 1)
            
            time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
            fixed_time = ref_time_stamps[time_idxs]
            pe_coord1 = torch.ones_like(coord[:, :, -1].unsqueeze(2)) * prev_time
            pe_coord = torch.ones_like(coord[:, :, -1].unsqueeze(2)) * time 
            # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * fixed_time
            # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
            # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
            # bp()
            mask_inp = torch.cat([q_feat, grid.flip(-1), pe_coord], dim=-1)
            mask_map = F.sigmoid(F.fold(self.mask_imnet(mask_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=1, padding=0) * 8)
            ones_map = torch.ones_like(mask_map)
            zeros_map = torch.zeros_like(mask_map)
            mask_map = torch.where(mask_map > 0, ones_map, zeros_map)
            mask = mask_map.view(bs, qs, -1) 
            encode_inp = torch.cat([q_feat * mask + (1 - mask) * q_feat1, pe_coord], dim=-1)
            # temporal_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1)
            # final_inp = torch.cat([q_feat, temporal_feat], dim=-1)
            # pred = F.fold(self.output_imnet(final_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=1, padding=0)
                
            # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
            # pred = self.decode_conv(output_feat)
            # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
            pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
            
#             flow_x, flow_y = torch.zeros_like(flow_t_to_0), torch.zeros_like(flow_t_to_0)
#             flow_x[:, :, 1:] = torch.abs(flow_t_to_0[:, :, 1:] - flow_t_to_0[:, :, :-1])
#             flow_x[:, :, 0] = flow_x[:, :, 1]
#             flow_y[:, :, :, 1:] = torch.abs(flow_t_to_0[:, :, :, 1:] - flow_t_to_0[:, :, :, :-1])
#             flow_y[:, :, :, 0] = flow_y[:, :, :, 1]
                
#             flow_grad = torch.sqrt((flow_x ** 2).sum(1)) + torch.sqrt((flow_y ** 2).sum(1))
#             norm_flow_dist = (flow_grad - flow_grad.min()) / (flow_grad.max() - flow_grad.min())
            # flow_dists.append(flow_grad)
            # norm_flow_dist = flow_x + flow_y
                
            # flow_dist = torch.sqrt(flow_t_to_0[:, 0] ** 2 + flow_t_to_0[:, 1] ** 2)
            # norm_flow_dist = (flow_dist - flow_dist.min()) / (flow_dist.max() - flow_dist.min())
            # flow_dists.apppend(norm_flow_dist)
            # return pred + self.mean_.squeeze(2)
            return pred, flow_map1.unsqueeze(1), mask_map
        
    def generate_flow_grad(self, flow_map, coord, flag=False, hard=False):
        with torch.no_grad():
            # flow_map = F.interpolate(flow_map, scale_factor=0.125, mode='bilinear', align_corners=True)
            # flow_map = flow_map / 8
            map_size = max((int(torch.sqrt((flow_map ** 2).sum(1)).max()) * 2) + 1, 5)
            map_size = min(49, map_size)
            map_size = int(map_size)
            
            flow_x, flow_y = torch.zeros_like(flow_map), torch.zeros_like(flow_map)
            flow_x[:, :, 1:] = torch.abs(flow_map[:, :, 1:] - flow_map[:, :, :-1])
            flow_x[:, :, 0] = flow_x[:, :, 1]
            flow_y[:, :, :, 1:] = torch.abs(flow_map[:, :, :, 1:] - flow_map[:, :, :, :-1])
            flow_y[:, :, :, 0] = flow_y[:, :, :, 1]
                
            flow_grad = torch.sqrt((flow_x ** 2).sum(1)) + torch.sqrt((flow_y ** 2).sum(1))
            del flow_x, flow_y
            torch.cuda.empty_cache() 
            flow_grad = F.unfold(flow_grad.unsqueeze(1), map_size, padding=(map_size // 2)).view(flow_grad.shape[0], map_size ** 2, flow_grad.shape[1], flow_grad.shape[2])
            flow_grad = flow_grad.max(1)[0]
            # flow_grad = F.interpolate(flow_grad.unsqueeze(1), scale_factor=8, mode='bilinear', align_corners=True)[:, 0]
            flow_grad = (flow_grad - flow_grad.min()) / (flow_grad.max() - flow_grad.min() + 1e-6)
            
            q_flow = F.grid_sample(
                        flow_grad.unsqueeze(1), coord.flip(-1).unsqueeze(1),
                        mode='nearest', padding_mode="border", align_corners=True)[:, :, 0, :] \
                        .permute(0, 2, 1)
            
            if flag == True:
                print(map_size)
            
            if hard == True:
                mean_value = flow_grad.mean()
                flow_grad[flow_grad > mean_value] = 1
                flow_grad[flow_grad <= mean_value] = 0
                q_flow[q_flow > mean_value] = 1
                q_flow[q_flow <= mean_value] = 0
                
            return flow_grad, q_flow
        
    def forward(self, inp, coord, Training=True, index=0, gts=None):
        self.gts = gts
        self.inp = inp
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

def convert_to_gray(img):
    gray_img = img[:, 0] * 0.299 + img[:, 1] * 0.587 + img[:, 2] * 0.114
    return gray_img.unsqueeze(1)

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
        
def get_raft_args():
    parser_raft = argparse.ArgumentParser()
    parser_raft.add_argument('--model', default='/model/nnice1216/video/raft-small.pth', help="restore checkpoint")
    parser_raft.add_argument('--path', default='/data/nnice1216/vimeo_septuplet/DAVIS/JPEGImages/Full-Resolution/bmx-rider/', help="dataset for evaluation")
    parser_raft.add_argument('--small', default=True, help='use small model')
    parser_raft.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser_raft.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args1 = parser_raft.parse_known_args()[0]
    
    return args1


def make_coord_2d(shape, ranges=None, flatten=False):
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