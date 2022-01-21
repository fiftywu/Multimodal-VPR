import os
import torch
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image
import random

DS_full2train = {4:1,10:1,
                 0:0,1:0,2:0,3:0,5:0,6:0,7:0,8:0,9:0,11:0,12:0}
SS_full2train = {0:0,
                 1:1,
                 2:2,
                 3:3,4:3,5:3,10:3,11:3,12:3,
                 6:4,
                 7:5,
                 8:6,
                 9:7}
# SS: 0:background; 1:buildings;2:plants;3:roadline;4:road;5:sidewalk;

train2color = {0: (128,128,128), 1:(128, 0, 0), 2:(64, 64, 128), 3:(0, 0, 0), 4:(0, 128, 64),
               5:(128, 64, 128), 6:(0, 0, 192), 7:(192, 192, 0)}

class DataProcess(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(DataProcess, self).__init__()
        self.pair_paths = sorted([os.path.join(opt.dataroot, opt.phase, name)
                           for name in os.listdir(os.path.join(opt.dataroot, opt.phase))])
        self.opt = opt

    def __getitem__(self, index):
        ABC_path = self.pair_paths[index]
        ABC = Image.open(ABC_path) #PIL 0,255
        # split ABC image into A and B
        w, h = ABC.size
        w2 = int(w/3)
        A = ABC.crop((0, 0, w2, h))
        B = ABC.crop((w2, 0, w2*2, h))
        C = ABC.crop((w2*2, 0, w, h))

        transform_paras = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_paras, grayscale=False)
        B_transform = get_transform(self.opt, transform_paras, grayscale=False)
        C_transform = get_transform(self.opt, transform_paras, grayscale=False)

        DI2 = A_transform(A)
        SI2 = B_transform(B)


        return {'DI': DI2,
                'DS': DI2,
                'SI': SI2,
                'SS': SI2}

    def __len__(self):
        return len(self.pair_paths)


def get_params(opt, each_size):
    w, h = each_size
    new_h, new_w = h, w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip_lr = random.random() > 0.5
    flip_td = False
    return {'crop_pos': (x, y), 'flip_lr': flip_lr, 'flip_td': flip_td}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip_lr']:
            transform_list.append(transforms.Lambda(lambda img: __flip_lr(img, params['flip_lr'])))
        elif params['flip_td']:
            transform_list.append(transforms.Lambda(lambda img: __flip_td(img, params['flip_td'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __flip_lr(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __flip_td(img, flip):
    if flip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
