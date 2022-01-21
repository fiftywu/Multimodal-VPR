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
        os.makedirs(result_dir)

    S_PAs, S_MAs, S_MIOUs, S_FWIOUs = [np.zeros([1, 8])] * 4
    with torch.no_grad():
        start_time = time.time()
        for COUNT, sample in tqdm.tqdm(enumerate(iterator_test)):
            visuals, error = model.test(sample)
            S_PAs = np.vstack([S_PAs, error['S_PA']])
            S_MAs = np.vstack([S_MAs, error['S_MA']])
            S_MIOUs = np.vstack([S_MIOUs, error['S_MIOU']])
            S_FWIOUs = np.vstack([S_FWIOUs, error['S_FWIOU']])
            save_visuals(visuals, opt, COUNT)
        print('time per sample', (time.time()-start_time) / dataset.__len__())
        print('S_PA', np.nanmean(np.nanmean(S_PAs[1:], 0)), '\n'
              'S_MA', np.round(np.nanmean(S_MAs[1:], 0), 4), np.nanmean(np.nanmean(S_MAs[1:], 0)), '\n',
              'S_MIOU', np.round(np.nanmean(S_MIOUs[1:], 0), 4), np.nanmean(np.nanmean(S_MIOUs[1:], 0)), '\n'
              'S_FWIOU', np.round(np.nanmean(S_FWIOUs[1:], 0), 4), np.nansum(np.nanmean(S_FWIOUs[1:], 0)))

def save_visuals(visuals, opt, COUNT):
    des_dir = os.path.join('visuals', opt.name, opt.phase, 'epoch_'+str(opt.epoch))
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    c, h, w = visuals['real_DI'].size()
    real_DI = ((visuals['real_DI'] + 1) / 2. * 255).permute(1,2,0).cpu().numpy()
    real_SI = ((visuals['real_SI'] + 1) / 2. * 255).permute(1,2,0).cpu().numpy()
    real_SS = ((visuals['real_SS'] + 1) / 2. * 255).permute(1,2,0).cpu().numpy()
    fake_SS = ((visuals['fake_SS'] + 1) / 2. * 255).permute(1,2,0).cpu().numpy()

    cv2.imwrite(os.path.join(des_dir, ("%d" % COUNT)+'_real_DI'+'.png'), real_DI)
    cv2.imwrite(os.path.join(des_dir, ("%d" % COUNT)+'_real_SI'+'.png'), real_SI)
    cv2.imwrite(os.path.join(des_dir, ("%d" % COUNT)+'_real_SS'+'.png'), real_SS)
    cv2.imwrite(os.path.join(des_dir, ("%d" % COUNT)+'_fake_SS'+'.png'), fake_SS)

if __name__ == '__main__':
    test_one_epoch()
