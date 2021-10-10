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

from models.base_conv_gru import *
from models.ode_func import ODEFunc, DiffeqSolver
from models.layers import create_convnet

@register('liif_ode')
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
        
        imnet_in_dim2 = 577
        
        imnet_spec2 = {'name': 'mlp', 'args': {'out_dim': 27, 'hidden_list': [256, 64]}}
        # self.encode_imnet = models.make(imnet_spec2, args={'in_dim': imnet_in_dim2})
        self.encode_imnet = Siren(in_features=imnet_in_dim2, out_features=27, hidden_features=[64, 64, 256, 256, 256], 
                  hidden_layers=3, outermost_linear=True)
        
        self.encode_imnet_ode = Siren(in_features=imnet_in_dim2, out_features=27, hidden_features=[64, 64, 256, 256], 
                  hidden_layers=3, outermost_linear=True)
        
        ode_func_netE = create_convnet(n_inputs=ode_dim,
                                       n_outputs=base_dim,
                                       n_layers=self.opt.n_layers,
                                       n_units=base_dim // 2).to(self.device)
        
        rec_ode_func = ODEFunc(opt=self.opt,
                               input_dim=ode_dim,
                               latent_dim=base_dim,  # channels after encoder, & latent dimension
                               ode_func_net=ode_func_netE,
                               device=self.device).to(self.device)
        
        z0_diffeq_solver = DiffeqSolver(base_dim,
                                        ode_func=rec_ode_func,
                                        method="euler",
                                        latents=base_dim,
                                        odeint_rtol=1e-3,
                                        odeint_atol=1e-4,
                                        device=self.device)
        
        self.encoder_z0 = Encoder_z0_ODE_ConvGRU(input_size=input_size,
                                                 input_dim=base_dim,
                                                 hidden_dim=base_dim,
                                                 kernel_size=(3, 3),
                                                 num_layers=1,
                                                 dtype=torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor,
                                                 batch_first=True,
                                                 bias=True,
                                                 return_all_layers=True,
                                                 z0_diffeq_solver=z0_diffeq_solver,
                                                 run_backwards=self.opt.run_backwards).to(self.device)      
        
        
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
        
        # self.mean_ = inp.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        # inp = inp - self.mean_ 
        # with torch.no_grad():
        self.feat = self.encoder(inp)
        truth_time_steps = [1, 2, 3, 4]
        bp()
        first_point_mu, first_point_std = self.encoder_z0(input_tensor=self.feat, time_steps=truth_time_steps, mask=None)
        
        return 

    def query_rgb(self, coord, Training=True):
        feat = self.feat
        
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        
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
        example_shape = make_coord_2d((feat.shape[2], feat.shape[3]))[None].cuda()
        if Training == True:
            for c in range(3):
                bs, qs = coord[c].shape[0], example_shape.shape[1]
                coord_ = example_shape.repeat(bs, 1, 1).clone()
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                time = coord[c][0, 0, -1].item()
                interval = coord[c][0, 0, -2].item()
                time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
                pe_coord = torch.ones_like(example_shape[:, :, 0].repeat(bs, 1).unsqueeze(2)) * time
                pe_coord2 = torch.ones_like(example_shape[:, :, 0].repeat(bs, 1).unsqueeze(2)) * (time - 1)
                # pe_coord2 = torch.ones_like(coord[c][:, :, -1].unsqueeze(2)) * interval
                # pe_coord = self.position_encoding(time)[:, 0].repeat(q_feat.shape[0], q_feat.shape[1], 1).cuda()
                encode_inp = torch.cat([q_feat, pe_coord], dim=-1)
                
                # output_feat = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 64).permute(0, 2, 1).view(bs, 64, feat.shape[2], feat.shape[3])
                # pred = self.decode_conv(output_feat)
                # pred = self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, 3)
                pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1))[1].view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
                # pred = pixel_shuffle(pred, 4)
                # pred = F.fold(self)
                # preds.append(pred + self.mean_.squeeze(2))
                preds.append(pred)
                
                
            return preds
        
        elif Training == "testing":
            for c in range(7):
                
                bs, qs = coord[c].shape[0], example_shape.shape[1]
                coord_ = example_shape.repeat(bs, 1, 1).clone()
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                time = coord[c][0, 0, -1].item()
                pe_coord = torch.ones_like(example_shape[:, :, 0].unsqueeze(2)) * time
                encode_inp = torch.cat([q_feat, pe_coord], dim=-1)
                
                pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1))[1].view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
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
                    q_feat = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                
                    pe_coord = torch.ones_like(example_shape[:, :, 0].repeat(bs, 1).unsqueeze(2)) * time
                    encode_inp = torch.cat([q_feat, pe_coord], dim=-1)

                    pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1)).view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2] * 2, feat.shape[3] * 2), kernel_size=3, padding=1)
                    preds.append(pred)

            return preds

        else:
            
            bs, qs = coord.shape[0], example_shape.shape[1]
            coord_ = example_shape.repeat(bs, 1, 1).clone()
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
            q_feat = F.grid_sample(
                feat, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            
            time = coord[0, 0, -1].item()
            time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
            pe_coord = torch.ones_like(example_shape[:, :, 0].unsqueeze(2)) * time
                
            encode_inp = torch.cat([q_feat, pe_coord], dim=-1)
                
            pred = F.fold(self.encode_imnet(encode_inp.view(bs * qs, -1))[1].view(bs, qs, -1).permute(0, 2, 1), output_size=(feat.shape[2], feat.shape[3]), kernel_size=3, padding=1)
            return pred
        

    def forward(self, inp, coord, Training=True):
        self.gen_feat(inp, Training)
        return self.query_rgb(coord, Training)
    
    
    
def make_coord_2d(shape, ranges=None, flatten=True):
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