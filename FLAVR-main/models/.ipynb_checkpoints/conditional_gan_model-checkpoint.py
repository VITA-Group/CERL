import random
import numpy as np
import torch
import os
from collections import OrderedDict
import torch.nn as nn
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss
from .models import make
from pdb import set_trace as bp

try:
    xrange          # Python2
except NameError:
    xrange = range  # Python 3
    

class ConditionalGAN(BaseModel):
    def name(self):
        return 'ConditionalGANModel'

    def __init__(self, opt):
        super(ConditionalGAN, self).__init__(opt)
        self.opt = opt
        self.isTrain = opt.isTrain
        
        self.gpu_ids = [Id for Id in range(torch.cuda.device_count())]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # define tensors
        # self.input_A = self.Tensor(opt.batchSize, opt.input_nc,  opt.fineSize, opt.fineSize)
        # self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # load/define networks
        # Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
        use_parallel = not opt.gan_type == 'wgan-gp'
        print("Use Parallel = ", "True" if use_parallel else "False")
        model_args = {'encoder_spec': {'name': 'edsr-baseline', 'args': {'no_upsampling': True}}, 'imnet_spec': {'name': 'mlp', 'args': {'out_dim': 3, 'hidden_list': [64, 64]}}}
        model_spec = {'name': 'liif_den', 'args': model_args}
        model = make(model_spec).to(self.device)
        self.netG = nn.DataParallel(model, device_ids=self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.gan_type == 'gan'
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.which_model_netD,
                opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel
            )
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam( self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )
            self.optimizer_D = torch.optim.Adam( self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )

            # self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1
            self.criticUpdates = 1

            # define loss functions
            self.discLoss, self.contentLoss = init_loss(opt)
        '''
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')
        '''

    def set_input(self, input):
        self.input_A = input['A']
        self.input_B = input['B']

    def forward(self):
        self.fake_B = self.netG(self.input_A[0], self.input_A[1], True)
        self.real_B = Variable(self.input_B)
        self.fake_B = Variable(torch.stack(self.fake_B, dim=1))
        
    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.input_A[0], self.input_A[1])
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        self.loss_D = self.discLoss.get_loss(self.netD, self.fake_B, self.real_B)

        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        self.loss_G_GAN = self.discLoss.get_g_loss(self.netD, self.fake_B)
        # Second, G(A) = B
        self.loss_G_Content = self.contentLoss.get_loss(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_Content

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        
        for iter_d in xrange(self.criticUpdates):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_Content.item()),
                            ('D_real+fake', self.loss_D.item())
                            ])
    '''
    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])
    '''

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr