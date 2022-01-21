import torch
import random
import torch.nn.functional as F
from .base_model import BaseModel
from .loss import VGG16, PerceptualLoss, StyleLoss
from . import networks
import numpy as np
from .segLoss import SegMetrics
import matplotlib.pyplot as plt
from .InpaintingScripts import ImageInpaintingMetric

"""
load segmentation model
use emptycities dataset to train image translation model
"""

class SegTransNet(BaseModel):
    def __init__(self, opt):
        super(SegTransNet, self).__init__(opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        self.vgg = VGG16()
        self.PerceoptualLoss = PerceptualLoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.num_classes = 8
        weight = np.array([0.3725, 0.0548, 0.0114, 0.0210, 0.0112, 0.3457, 0.0859, 0.0975], dtype=np.float32)
        weight = 1. / np.log(weight * 100 + 1.02)
        weight = torch.tensor(weight).to(self.device)
        self.criterionBCE = torch.nn.NLLLoss2d(weight)
        self.netG = networks.define_G(input_nc=1, output_nc=1, ngf=32*1, netG='EasyNet', norm=opt.norm,
                                      use_dropout=not opt.no_dropout, init_type=opt.init_type,
                                      init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        '''
        load the param for the coarse_net
        '''
        SegNet_param = torch.load('checkpoints/Seg0221_ERFNet_W2_BZ1/60_net_G.pth',
                                     map_location=lambda storage, loc: storage.cuda(opt.gpu_ids[0]))
        for params_name, params in self.netG.module.netS.named_parameters():
            if params_name in self.netG.module.netS.state_dict():
                self.netG.module.netS.state_dict()[params_name].copy_(SegNet_param['net'][params_name])

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        if self.isTrain:
            self.netD = networks.define_D(input_nc=1+1, ndf=32*1, n_layers_D=4, netD='basic', norm=opt.norm,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.optimizers = []
            netG_S = self.netG.module.netS.parameters()
            ignored_parameters = list(map(id, self.netG.module.netS.parameters()))
            base_parameters = filter(lambda p: id(p) not in ignored_parameters, self.netG.parameters())
            netG_param = [{'params': base_parameters, 'lr': opt.lr},
                          {'params': netG_S, 'lr': opt.lr*0.00}]
            self.optimizer_G = torch.optim.Adam(netG_param, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr*2., betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if self.isTrain:
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def set_input(self, samples):
        self.real_DS = samples['DS'].to(self.device)
        self.real_DI = samples['DI'].to(self.device)
        self.real_SS = samples['SS'].to(self.device)
        self.real_SI = samples['SI'].to(self.device)

    def rgb2gray(self, x):
        # x [n, 3, h, w]
        x_gray = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
        x_gray = x_gray.view(x.shape[0], 1, x.shape[2], x.shape[3])
        return x_gray

    def forward(self):
        self.fake_SS,  self.fake_SI= self.netG(self.real_DI)

    def backward_D(self):
        # fake
        fake = torch.cat((self.rgb2gray(self.real_DI), self.fake_SI), 1)
        pred_fake = self.netD(fake.detach())
        # real
        real = torch.cat((self.rgb2gray(self.real_DI), self.rgb2gray(self.real_SI)), 1)
        pred_real = self.netD(real.detach())
        # GAN LOSS
        self.loss_D_real = torch.mean((pred_real - 1) ** 2)
        self.loss_D_fake = torch.mean((pred_fake - 0) ** 2)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # GAN loss
        fake = torch.cat((self.rgb2gray(self.real_DI), self.fake_SI), 1)  # nothing with S
        pred_fake = self.netD(fake)
        self.loss_G_GAN = torch.mean((pred_fake - 1.) ** 2)

        # L1 loss
        self.loss_G_L1 = self.criterionL1(self.rgb2gray(self.real_SI), self.fake_SI) * 40.

        # Seg loss
        # b, c, h, w = self.real_SS.size() #b, 8, h, w
        # self.loss_G_SS = self.criterionBCE(
        #     torch.log_softmax(self.fake_SS, dim=1),
        #     self.real_SS.permute(0, 2, 3, 1).view(b, -1, c).max(2)[1].view(b, h, w)) * 1.

        # perception loss
        self.loss_G_P = self.PerceoptualLoss(torch.cat([self.rgb2gray(self.real_SI)]*3, 1),
                                             torch.cat([self.fake_SI]*3, 1)) * 1.

        # joint loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_P
        self.loss_G.backward()

    def optmize_parameters(self):
        # compute fake images: G(A)
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def get_current_loss(self):
        loss = dict()
        loss['G_GAN'] = self.loss_G_GAN.item()
        loss['G_L1'] = self.loss_G_L1.item()
        loss['G_P'] = self.loss_G_P.item()
        loss['G'] = self.loss_G.item()
        loss['D_real'] = self.loss_D_real.item()
        loss['D_fake'] = self.loss_D_fake.item()
        loss['D'] = self.loss_D.item()
        return loss

    def onehot2rgb(self, output):
        device = output.device
        train2color = {0: (128, 128, 128), 1: (128, 0, 0), 2: (64, 64, 128), 3: (0, 0, 0), 4: (0, 128, 64),
                       5: (128, 64, 128), 6: (0, 0, 192), 7: (192, 192, 0)}
        # output = F.log_softmax(output)
        c, h, w = output.size()
        b = 1
        output = output.view(1, c, h, w)
        pred = output.permute(0, 2, 3, 1).view(-1, self.num_classes).max(1)[1].view(b, 1, h, w)
        pred_color = torch.zeros(b, 3, h, w)
        for k, v in train2color.items():
            pred_r = torch.zeros(b, 1, h, w)
            pred_r[pred == k] = v[0]
            pred_g = torch.zeros(b, 1, h, w)
            pred_g[pred == k] = v[1]
            pred_b = torch.zeros(b, 1, h, w)
            pred_b[pred == k] = v[2]
            pred_color += torch.cat((pred_r, pred_g, pred_b), 1)
        return (pred_color.to(device) / 255. - 0.5) * 2

    def onehot2map(self, output):
        c, h, w = output.size()
        output = output.view(1, c, h, w)
        b = 1
        pred = output.permute(0, 2, 3, 1).view(-1, self.num_classes).max(1)[1].view(b, 1, h, w)
        return torch.cat(([pred]*3))

    def get_current_visuals(self):
        visuals = dict()
        visuals['real_DI'] = self.real_DI[0]
        visuals['real_SI'] = self.real_SI[0]
        # visuals['real_SS'] = self.onehot2rgb(self.real_SS[0])[0]
        visuals['fake_SI'] = torch.cat([self.fake_SI[0]]*3, 0)
        visuals['fake_SS'] = self.onehot2rgb(self.fake_SS[0])[0]
        return visuals

    def get_statistic_errors(self):
        errors = dict()
        fake_SI = (self.fake_SI + 1) * 0.5
        real_SI = (self.rgb2gray(self.real_SI) + 1) * 0.5
        Metric = ImageInpaintingMetric(fake_SI[0][0].detach().cpu().numpy(), real_SI[0][0].detach().cpu().numpy())
        Metric.evaluate()
        errors['L1'] = Metric.l1
        errors['L2'] = Metric.l2
        errors['PSNR'] = Metric.psnr
        errors['SSIM'] = Metric.ssim
        # errors['S_PA'], errors['S_MA'], errors['S_MIOU'], errors['S_FWIOU'] = self.iou(self.real_SS[0], self.fake_SS[0])
        return errors

    def test(self, samples):
        self.set_input(samples)
        self.forward()
        visuals, error = [], []
        visuals = self.get_current_visuals()
        error = self.get_statistic_errors()
        return visuals, error

    def iou(self, target, pred):
        # input
        # target,pred onehot B,C,H,W

        # pred numpy HXW
        # target numoy HXW
        # nclasses = 8

        ## step 1 onehot2map
        c, h, w = target.size()
        b = 1
        target = target.view(b, c, h, w)
        target = target.permute(0, 2, 3, 1).view(-1, c).max(1)[1].view(h, w)
        pred = pred.view(b, c, h, w)
        pred = pred.permute(0, 2, 3, 1).view(-1, c).max(1)[1].view(h, w)
        Metrics = SegMetrics(target.cpu().numpy(), pred.cpu().numpy(), c)
        ius = Metrics.perclass_iu()
        return Metrics.mean_metric()

