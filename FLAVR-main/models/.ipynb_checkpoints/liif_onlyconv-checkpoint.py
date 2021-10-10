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


@register('liif_onlyconv')
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
        
        imnet_spec2 = {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [256, 256, 256]}}
        # self.encode_imnet = models.make(imnet_spec2, args={'in_dim': imnet_in_dim2})
        self.encode_imnet = Siren(in_features=imnet_in_dim2, out_features=3, hidden_features=256, 
                   hidden_layers=3, outermost_linear=True)
        self.outconv = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, 21 , kernel_size=7 , stride=1, padding=0) 
            )
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

        inp = torch.stack(inp, dim=2)
        self.h = inp.shape[3]
        self.w = inp.shape[4]
        
        # self.mean_ = inp.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        # inp = inp - self.mean_ 
        # with torch.no_grad():
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None, Training=True):
        feat = self.outconv(self.feat)
        preds = []
        if not Training == False:
            for i in range(7):
                pred = feat[:, i * 3 : i * 3 + 3]
                preds.append(pred)
                
            return preds
    
        else:
            ref_time_stamps = np.array([i / 8 for i in range(1, 8)])
            time = coord[0, 0, -1].item()
            time_idxs = np.argsort(np.abs(ref_time_stamps - time))[0]
            pred = feat[:, time_idxs * 3 : time_idxs * 3 + 3]
            
            return pred
             

    def forward(self, inp, coord, cell, Training=True):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell, Training)