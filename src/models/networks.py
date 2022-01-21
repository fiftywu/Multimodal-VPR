import os
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchfcn
from models.erfnet import Net

###############################################################################
# Helper Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions锛庛€€
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], init=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init:
        init_weights(net, init_type, init_gain=init_gain)
    return net


class SSNet(nn.Module):
    def __init__(self, input_nc, output_nc, nsf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SSNet, self).__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_nc, nsf, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(nsf),
            self.act
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(nsf, nsf * 2, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(nsf * 2),
            self.act
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(nsf * 2, nsf, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(nsf),
            self.act
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(nsf, output_nc, kernel_size=4, stride=2, padding=1, bias=True),
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        return x

def define_S(input_nc, output_nc, netS='SSNet', nsf=64, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], outputACT=True):
    norm_layer = get_norm_layer(norm_type=norm)
    if netS == 'SSNet':
        net = SSNet(input_nc, output_nc, nsf=nsf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netS == 'unet8':
        net = UnetGenerator(input_nc, output_nc, 8, ngf=nsf, norm_layer=norm_layer, use_dropout=use_dropout, outputACT=outputACT)
    else:
        print('Not Found-->', netS)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- name of netG
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'unet8':
        net = UnetGenerator(input_nc, output_nc, 8, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        # net = UnetGenerator(input_nc, output_nc, 6, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    if netG == 'SegTransNet':
        net = SegTransNet(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    if netG == 'SegNet':
        net = UnetGenerator(input_nc, output_nc, 8, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, outputACT=False)
    if netG == 'FCN32s':
        net = torchfcn.models.FCN32s(n_class=output_nc)
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        net.copy_params_from_vgg16(vgg16)
        return init_net(net, init_type, init_gain, gpu_ids, init=False)
    if netG == 'FCN16s':
        pass
    if netG == 'ERFNet':
        from models.erfnet import Net
        net = Net(8)
    if netG == 'UResNet':
        net = UResNet(input_nc, output_nc, ngf=ngf, norm_layer=nn.BatchNorm2d)
    if netG == 'MGAN':
        net = MGAN(input_nc, output_nc, ngf=ngf, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
    if netG == 'OURS':
        net = OURS(input_nc, output_nc, ngf=ngf, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
    if netG == 'EasyNet':
        net = EasyNet(input_nc, output_nc, ngf=ngf, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
    if netG == 'ComplexNet':
        net = ComplexNet(input_nc, output_nc, ngf=ngf, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
    else:
        print('Not Found-->', netG)

    return init_net(net, init_type, init_gain, gpu_ids)

# Define the resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class UResNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(UResNet, self).__init__()
        res_num = 4
        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf*8, 1)
            blocks.append(block)

        self.down1 = conv_block(input_nc, ngf, 'start', norm_layer=norm_layer)  # x1=[128,128,64]
        self.down2 = conv_block(ngf, ngf * 2, 'down', norm_layer=norm_layer)  # x2=[64,64,128]
        self.down3 = conv_block(ngf * 2, ngf * 4, 'down', norm_layer=norm_layer)  # x3=[32,32,256]
        self.down4 = conv_block(ngf * 4, ngf * 8, 'down', norm_layer=norm_layer)  # x4=[16,16,512]
        self.down5 = conv_block(ngf * 8, ngf * 8, 'buttom', norm_layer=norm_layer)  # x4=[16,16,512]
        self.res = nn.Sequential(*blocks)
        self.up4 = conv_block(ngf * 16, ngf * 8, 'up', norm_layer=norm_layer, use_dropout=True)  # x15=[16,16,512]
        # x16 = [x4, x15] = [16, 16, 1024]
        self.up5 = conv_block(ngf * 16, ngf * 4, 'up', norm_layer=norm_layer, use_dropout=True)  # x17=[32,32,256]
        # x18 = [x3, x17] = [32, 32, 512]
        self.up6 = conv_block(ngf * 8, ngf * 2, 'up', norm_layer=norm_layer)  # x19=[64,64,128]
        # x20 = [x2, x19] = [64, 64, 256]
        self.up7 = conv_block(ngf * 4, ngf, 'up', norm_layer=norm_layer)  # x21=[128,128,64]
        # x22 = [x1, x21] = [128, 128, 128]
        self.up8 = conv_block(ngf * 2, output_nc, 'end', norm_layer=norm_layer)  # x23=[256,256,3]

    def forward(self, x):
        x1 = self.down1(x) # 256 256 n
        x2 = self.down2(x1) # 128 128 2n
        x3 = self.down3(x2) # 64 64 4n
        x4 = self.down4(x3) # 32 32 8n
        x5 = self.down5(x4) # 16 16 8n
        xx = self.res(x5) # 16 16 8n
        x14 = torch.cat((xx, x5), dim=1)
        x15 = self.up4(x14) # 32 32 8n
        x16 = torch.cat((x4, x15), dim=1)
        x17 = self.up5(x16) # 64 64 4n
        x18 = torch.cat((x3, x17), dim=1)
        x19 = self.up6(x18) # 128 128 2n
        x20 = torch.cat((x2, x19), dim=1)
        x21 = self.up7(x20) # 256 256 n
        x22 = torch.cat((x1, x21), dim=1)
        x23 = self.up8(x22) # 512 512 3
        return x23

class conv_block(nn.Module):
    def __init__(self, inner_nc, outer_nc, part_name, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(conv_block, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if part_name is 'start':
            self.conv = nn.Sequential(
                nn.Conv2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            )

        elif part_name == 'buttom':
            self.conv = nn.Sequential(
                nn.Conv2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                nn.ReLU(True)
            )

        elif part_name == 'down':
            self.conv = nn.Sequential(
                nn.Conv2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(outer_nc),
                nn.LeakyReLU(0.2, True)
            )
        elif part_name == 'up':
            if use_dropout:
                self.conv = nn.Sequential(
                    nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(outer_nc),
                    nn.Dropout(0.5),
                    nn.ReLU(True)
                )
            else:
                self.conv = nn.Sequential(
                    nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(outer_nc),
                    nn.ReLU(True),
                )
        elif part_name == 'end':
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            )

        elif part_name == 'down_dilation':
            self.conv = nn.Sequential(
                # same as ksize = 5
                nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=2, padding=2, dilation=2, bias=use_bias),
                norm_layer(outer_nc),
                nn.Sigmoid()
            )

        elif part_name == 'up_dilation':
            self.conv = nn.Sequential(
                # same as ksize = 5
                nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, dilation=1, bias=use_bias),
                norm_layer(outer_nc),
                nn.Tanh()
            )
        else:
            pass

    def forward(self, x):
        x = self.conv(x)
        return x


    # self.SDA = nn.Sequential(*blocks)

class MGAN(nn.Module):
    def __init__(self, input_nc, output_nc, num_down=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MGAN, self).__init__()
        # Dynamic2static segmentation ---> ERFNet
        self.seg_net = Net(2)
        # Dynamic2static image translation ---> UNet
        # add 1 denotes dynamic mask
        self.static_net = UnetGenerator(input_nc+1, output_nc, num_down, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    def forward(self, x):
        fake_SS = self.seg_net(x)
        b, c, h, w = fake_SS.size()
        fake_DS_map = fake_SS.permute(0, 2, 3, 1).view(b, -1, c).max(2)[1].view(b, 1, h, w)
        dmask = (fake_DS_map.to(torch.float) / (c - 1) - 0.5) * 2  # (norm to -1,1) 1-> dynamic
        fake_SI = self.static_net(torch.cat((x, dmask), 1))
        return fake_SS, dmask, fake_SI


# modified in 2021.7.14
class EasyNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_down=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(EasyNet, self).__init__()
        # generate network (input:: rgb + edge map)
        self.netG = UnetGenerator(input_nc*8+8, output_nc, num_down, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        # 8 is nums of classes
        self.netS = Net(8)  # Net is ERFNet (input is 3channels, output is 8 channels)

    def forward(self, x):
        fake_SS = self.netS(x)
        # fake_SS_8c = F.log_softmax(fake_SS)
        fake_SS_8c = F.softmax(fake_SS, dim=1)
        # semantics = self.netS(x, encoder=True)
        # b, c, h, w = fake_SS.size()
        # fake_SS_map = fake_SS.permute(0, 2, 3, 1).view(b, -1, c).max(2)[1].view(b, 1, h, w)
        # fake_SS_1c = (fake_SS_map.to(torch.float) / (c - 1) - 0.5) * 2  # (norm to -1,1)
        x = self.rgb2gray(x)
        x = torch.cat([x]*8, 1)
        fake_SI = self.netG(torch.cat((x, fake_SS_8c), 1))
        return fake_SS, fake_SI

    def rgb2gray(self, x):
        # x [n, 3, h, w]
        x_gray = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
        x_gray = x_gray.view(x.shape[0], 1, x.shape[2], x.shape[3])
        return x_gray

    # def process_semantics(self, t):
    #     # given semantics [b, c, h, w]
    #     edge = torch.ByteTensor(t.size()).zero_()
    #     edge = edge.to(t.device)
    #     edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
    #     edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
    #     edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    #     edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    #     return edge.float()



class ComplexNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_down=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ComplexNet, self).__init__()
        ## ERFNet
        self.netS = Net(8)
        ## UResNet
        res_num = 4
        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf*8, 1)
            blocks.append(block)
        self.down1 = conv_block(input_nc+1, ngf, 'start', norm_layer=norm_layer)  # x1=[128,128,64]
        self.down2 = conv_block(ngf, ngf * 2, 'down', norm_layer=norm_layer)  # x2=[64,64,128]
        self.down3 = conv_block(ngf * 2, ngf * 4, 'down', norm_layer=norm_layer)  # x3=[32,32,256]
        self.down4 = conv_block(ngf * 4, ngf * 8, 'down', norm_layer=norm_layer)  # x4=[16,16,512]
        self.down5 = conv_block(ngf * 8, ngf * 8, 'buttom', norm_layer=norm_layer)  # x4=[16,16,512]
        self.res = nn.Sequential(*blocks)
        self.up4 = conv_block(ngf * 16+8, ngf * 8, 'up', norm_layer=norm_layer, use_dropout=True)  # x15=[16,16,512]
        self.up5 = conv_block(ngf * 16+8, ngf * 4, 'up', norm_layer=norm_layer, use_dropout=True)  # x17=[32,32,256]
        self.up6 = conv_block(ngf * 8+8, ngf * 2, 'up', norm_layer=norm_layer)  # x19=[64,64,128]
        self.up7 = conv_block(ngf * 4+8, ngf, 'up', norm_layer=norm_layer)  # x21=[128,128,64]
        self.up8 = conv_block(ngf * 2+8, output_nc, 'end', norm_layer=norm_layer)  # x23=[256,256,3]

    def forward(self, x):
        fake_SS = self.netS(x)
        encoder_feat = self.netS(x, True) # 4,8,64,64
        b, c, h, w = fake_SS.size()
        fake_SS_map = fake_SS.permute(0, 2, 3, 1).view(b, -1, c).max(2)[1].view(b, 1, h, w)
        fake_SS_1c = (fake_SS_map.to(torch.float) / (c - 1) - 0.5) * 2  # (norm to -1,1)
        # ---
        # x1 = self.down1(x) # 256 256 n
        x1 = self.down1(torch.cat((x,fake_SS_1c), dim=1)) # 256 256 n
        x2 = self.down2(x1) # 128 128 2n
        x3 = self.down3(x2) # 64 64 4n
        x4 = self.down4(x3) # 32 32 8n
        x5 = self.down5(x4) # 16 16 8n
        xx = self.res(x5) # 16 16 8n
        x14 = torch.cat((xx, x5, F.interpolate(encoder_feat, x5.shape[2:])), dim=1)
        x15 = self.up4(x14) # 32 32 8n
        x16 = torch.cat((x4, x15, F.interpolate(encoder_feat, x15.shape[2:])), dim=1)
        x17 = self.up5(x16) # 64 64 4n
        x18 = torch.cat((x3, x17, F.interpolate(encoder_feat, x17.shape[2:])), dim=1)
        x19 = self.up6(x18) # 128 128 2n
        x20 = torch.cat((x2, x19, F.interpolate(encoder_feat, x19.shape[2:])), dim=1)
        x21 = self.up7(x20) # 256 256 n
        x22 = torch.cat((x1, x21, F.interpolate(encoder_feat, x21.shape[2:])), dim=1)
        x23 = self.up8(x22) # 512 512 3
        return fake_SS, x23


class OURS(nn.Module):
    def __init__(self, input_nc, output_nc, num_down=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(OURS, self).__init__()
        # generate network
        self.netG = UnetGenerator(input_nc+1+1, output_nc, num_down, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        # 8 is nums of classes
        self.netS = Net(8) # Net is ERFNet
        # 2 is nums of dynamic or staic
        self.netDS = Net(2)

    def forward(self, x):
        fake_SS = self.netS(x)
        b, c, h, w = fake_SS.size()
        fake_SS_map = fake_SS.permute(0, 2, 3, 1).view(b, -1, c).max(2)[1].view(b, 1, h, w)
        fake_SS_1c = (fake_SS_map.to(torch.float) / (c - 1) - 0.5) * 2  # (norm to -1,1)

        seg_DS = self.netDS(x)
        b2, c2, h2, w2 = seg_DS.size()
        seg_DS_map = seg_DS.permute(0, 2, 3, 1).view(b2, -1, c2).max(2)[1].view(b2, 1, h2, w2)
        dmask = (seg_DS_map.to(torch.float) / (c2 - 1) - 0.5) * 2  # (norm to -1,1) 1-> dynamic

        fake_SI = self.netG(torch.cat((x, fake_SS_1c, dmask), 1))
        return fake_SS, seg_DS, dmask, fake_SI

class SegTransNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_down=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SegTransNet, self).__init__()
        self.netG = UnetGenerator(input_nc+1+1, output_nc, num_down, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        # 8 is nums of classes
        # self.netS = UnetGenerator(input_nc, 8, num_down, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, outputACT=False)
        self.netS = Net(8) # Net is ERFNet
        self.SDA = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf),
            nn.ReLU(),
            nn.Conv2d(ngf, 2*ngf, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(2*ngf),
            nn.ReLU(),
            nn.Conv2d(2*ngf, 4*ngf, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(4*ngf),
            nn.ReLU(),
            nn.Conv2d(4*ngf, 4*ngf, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(4*ngf)
        )
        # output 4n*64*64

    # def forward(self, x):
    #     #input: x is dynamic image
    #     #output: static semantics, semantics difference map, static image
    #     fake_SS = self.netS(x)
    #     b, c, h, w = fake_SS.size()
    #     fake_SS_map = fake_SS.permute(0, 2, 3, 1).view(b, -1, c).max(2)[1].view(b, 1, h, w)
    #     fake_SS_1c = (fake_SS_map.to(torch.float) / (c - 1) - 0.5) * 2  # (norm to -1,1)
    #     fake_SS_3c = torch.cat((fake_SS_1c, fake_SS_1c, fake_SS_1c), 1)
    #     SDA_map = get_SDA(self.SDA(x), self.SDA(fake_SS_3c))
    #     SDA_map = F.interpolate(SDA_map, [h, w]).to(x.device)
    #     # fake_SI = self.netG(torch.cat((x, fake_SS_map.to(torch.float), SDA_map), 1))
    #     fake_SI = self.netG(torch.cat((x, SDA_map), 1))
    #     return fake_SS, SDA_map, fake_SI

    # def forward(self, x):
    #     # SDA1 SDA2 DIfference
    #     #input: x is dynamic image
    #     #output: static semantics, semantics difference map, static image
    #     fake_SS = self.netS(x)
    #     b, c, h, w = fake_SS.size()
    #     fake_SS_map = fake_SS.permute(0, 2, 3, 1).view(b, -1, c).max(2)[1].view(b, 1, h, w)
    #     fake_SS_1c = (fake_SS_map.to(torch.float) / (c - 1) - 0.5) * 2  # (norm to -1,1)
    #     fake_SS_3c = torch.cat((fake_SS_1c, fake_SS_1c, fake_SS_1c), 1)
    #     SDA_map_dynamic = get_SDA(self.SDA(x), self.SDA(fake_SS_3c))
    #     SDA_map_dynamic = F.interpolate(SDA_map_dynamic, [h, w]).to(x.device)
    #     fake_SI = self.netG(torch.cat((x, fake_SS_map.to(torch.float), SDA_map_dynamic), 1))
    #     SDA_map_static = get_SDA(self.SDA(fake_SI), self.SDA(fake_SS_3c))
    #     SDA_map_static = F.interpolate(SDA_map_static, [h, w]).to(x.device)
    #     return fake_SS, fake_SI, SDA_map_dynamic, SDA_map_static

    def forward(self, x):
        # SDA1 SDA2 DIfference
        #input: x is dynamic image
        #output: static semantics, semantics difference map, static image
        fake_SS = self.netS(x)
        b, c, h, w = fake_SS.size()
        fake_SS_map = fake_SS.permute(0, 2, 3, 1).view(b, -1, c).max(2)[1].view(b, 1, h, w)
        fake_SS_1c = (fake_SS_map.to(torch.float) / (c - 1) - 0.5) * 2  # (norm to -1,1)
        fake_SS_3c = torch.cat((fake_SS_1c, fake_SS_1c, fake_SS_1c), 1)
        SDA_map_dynamic = get_SDA(self.SDA(x), self.SDA(fake_SS_3c))
        SDA_map_dynamic = F.interpolate(SDA_map_dynamic, [h, w]).to(x.device)
        fake_SI = self.netG(torch.cat((x, fake_SS_map.to(torch.float), SDA_map_dynamic), 1))
        SDA_map2 = get_SDA(self.SDA(x), self.SDA(fake_SI))
        SDA_map2 = F.interpolate(SDA_map2, [h, w]).to(x.device)
        SDA_map3 = get_SDA(self.SDA(fake_SS_3c), self.SDA(fake_SI))
        SDA_map3 = F.interpolate(SDA_map3, [h, w]).to(x.device)
        return fake_SS, fake_SI, SDA_map_dynamic, SDA_map2, SDA_map3

def get_SDA(x, y):
    b, c, h, w = x.size()  # b, 8, h, w
    x = x.permute(0, 2, 3, 1).view(b, -1, c)
    y = y.permute(0, 2, 3, 1).view(b, -1, c)
    diff = torch.abs(torch.sum(x*y, 2)) / torch.sqrt(torch.sum(x*x, 2)) / torch.sqrt(torch.sum(y*y, 2))
    SDA = -diff.view(b, 1, h, w) + 1.
    # difference is 1
    return SDA



def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70脳70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)


    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, outputACT=True):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, outputACT=outputACT)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, outputACT=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if outputACT:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


'''
used scripts of Coarse2fineNet
'''
def get_diff_mask(coarse_fake_B, real_A, dmask):
    # input all float
    # output bool
    delta = torch.abs(coarse_fake_B-real_A)  # [0,2]
    diff_mask = delta > 0.3
    # diff_mask_final = diff_mask.float()
    # filter
    filter_diff_mask = F.avg_pool2d(diff_mask.float(), kernel_size=5, stride=1)
    filter_diff_mask = F.interpolate(filter_diff_mask, size=(256, 256), mode='nearest')
    diff_mask_final = (filter_diff_mask > 0.45).float()
    return diff_mask_final


def get_fine_mask(diff_mask, dmask):
    # input: diff_mask(bool), dmask(float)
    # output: fine_mask(float)
    batch_size = dmask.shape[0]
    fine_mask = torch.add(diff_mask > 0, dmask > 0).float()  # m2=1. is dynamic/ hole
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    for batch_item in range(batch_size):
        src = fine_mask[batch_item, 0, :, :].view(256, 256).cpu().detach().numpy()
        fine_mask[batch_item, 0, :, :] = torch.tensor(cv2.dilate(src, kernel)).to(dmask.device)
    return fine_mask


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
