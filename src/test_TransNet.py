from models.models import create_model
from options.test_options import TestOptions
import os
from data.dataprocess import DataProcess
import torch
from torch.utils import data
import time
import numpy as np
import cv2
import tqdm


def test_one_epoch():
    opt = TestOptions().parse()
    model = create_model(opt)
    model.print_networks()
    model.load_networks(opt.epoch)
    if opt.eval:
        model.netG.eval()
    result_dir = os.path.join(opt.results_dir, opt.name)
    dataset = DataProcess(opt)
    iterator_test = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    L1s, L2s, PSNRs, SSIMs = [], [], [], []

    with torch.no_grad():
        start_time = time.time()
        for COUNT, sample in tqdm.tqdm(enumerate(iterator_test)):
            visuals, error = model.test(sample)
            L1s.append(error['L1'])
            L2s.append(error['L2'])
            PSNRs.append(error['PSNR'])
            SSIMs.append(error['SSIM'])
            save_visuals(visuals, opt, COUNT)
            if COUNT % 100 == 0:
                print(error)
                print('L1', np.mean(L1s), 'L2', np.mean(L2s), 'PSNR', np.mean(PSNRs), 'SSIM', np.mean(SSIMs))
        print('time per sample', (time.time()-start_time) / dataset.__len__())
        print('## L1', np.mean(L1s), 'L2', np.mean(L2s), 'PSNR', np.mean(PSNRs), 'SSIM', np.mean(SSIMs))


def save_visuals(visuals, opt, COUNT):
    des_dir = os.path.join('visuals', opt.name, opt.phase, 'epoch_'+str(opt.epoch))
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    c, h, w = visuals['real_DI'].size()
    real_DI = ((visuals['real_DI'] + 1) / 2. * 255).permute(1,2,0).cpu().numpy()
    real_SI = ((visuals['real_SI'] + 1) / 2. * 255).permute(1,2,0).cpu().numpy()
    # real_SS = ((visuals['real_SS'] + 1) / 2. * 255).permute(1,2,0).cpu().numpy()
    # real_SS = visuals['real_SS'].permute(1,2,0).cpu().numpy()
    fake_SI = ((visuals['fake_SI'] + 1) / 2. * 255).permute(1,2,0).cpu().numpy()
    fake_SS = ((visuals['fake_SS'] + 1) / 2. * 255).permute(1,2,0).cpu().numpy()

    cv2.imwrite(os.path.join(des_dir, ("%d" % COUNT)+'_real_DI'+'.png'), real_DI)
    cv2.imwrite(os.path.join(des_dir, ("%d" % COUNT)+'_real_SI'+'.png'), real_SI)
    # cv2.imwrite(os.path.join(des_dir, ("%d" % COUNT)+'_real_SSM'+'.png'), real_SS)
    cv2.imwrite(os.path.join(des_dir, ("%d" % COUNT)+'_fake_SI'+'.png'), fake_SI)
    cv2.imwrite(os.path.join(des_dir, ("%d" % COUNT)+'_fake_SSM'+'.png'), fake_SS)



def test_each_epoch():
    opt = TestOptions().parse()
    for EPOCH in range(38, 150, 1):
        opt.epoch = EPOCH
        model = create_model(opt)
        model.load_networks(opt.epoch)
        if opt.eval:
            model.netG.eval()
        result_dir = os.path.join(opt.results_dir, opt.name)
        dataset = DataProcess(opt)
        iterator_test = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        L1s = []
        L2s, PSNRs, SSIMs = [], [], []
        with torch.no_grad():
            start_time = time.time()
            for COUNT, sample in tqdm.tqdm(enumerate(iterator_test)):
                visuals, error = model.test(sample)
                L1s.append(error['L1'])
                L2s.append(error['L2'])
                PSNRs.append(error['PSNR'])
                SSIMs.append(error['SSIM'])
                # save_visuals(visuals, opt, COUNT)
                if COUNT==400:
                    break
                # save_visuals(visuals, opt, COUNT)
        print('time per sample', (time.time() - start_time) / dataset.__len__())
        print('EPOCH->>',EPOCH,  'L1', np.mean(L1s), 'L2', np.mean(L2s), 'PSNR', np.mean(PSNRs), 'SSIM', np.mean(SSIMs))

if __name__ == '__main__':
    test_one_epoch()
    # test_each_epoch()
