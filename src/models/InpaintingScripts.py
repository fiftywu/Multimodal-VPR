import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
from scipy.signal import convolve2d
import cv2
# import skimage.measure


class ImageInpaintingMetric():
    def __init__(self, im1, im2, hole=0):
        # im1, im2, 0~1, float, channel=3
        self.im1 = im1
        self.im2 = im2
        self.hole = hole
        
    def evaluate(self):
        # overall / hole / unhole
        self.l1 = self.get_l1()
        self.l2 = self.get_l2()
        self.ssim = self.get_ssim()
        self.psnr = self.get_psnr()
    
    def get_l1(self):
        return np.mean(np.abs(self.im1-self.im2))
    
    def get_l2(self):
        return np.mean((self.im1-self.im2)**2)
        
    def get_psnr(self, L=1):
        mse = self.get_l2()
        if mse==0:
            return np.array([40])
        return 10 * np.log10(L * L / mse)
        
    def get_ssim(self, L=1):
        # im1_gray = cv2.cvtColor(self.im1, cv2.COLOR_RGB2GRAY)
        # im2_gray = cv2.cvtColor(self.im2, cv2.COLOR_RGB2GRAY)
        im1_gray = self.im1
        im2_gray = self.im2
        overall = compute_ssim(im1_gray, im2_gray, self.hole, L=L)
        return np.mean([overall])
    
    def printMeric(self):
        self.evaluate()
        print('L1 [overall/ hole/ non-hole] : ', self.l1,'\n', 
              'L2 [overall/ hole/ non-hole] : ', self.l2,'\n', 
              'SSIM [overall/ hole/ non-hole] : ', self.ssim,'\n', 
              'PSNR [overall/ hole/ non-hole] : ', self.psnr)

        
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, dmask=0, k1=0.01, k2=0.03, win_size=11, L=255, nomask=True):

    ## first convert im1, im2 to grayscale i.e., channel==1

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    if nomask:
        return np.mean(ssim_map)

    # if add a dmask, how to calculate the dmaks region
    dmask = cv2.resize(dmask, dsize=mu1.shape, interpolation=cv2.INTER_NEAREST)
    if np.sum(dmask)==0:
        hole_ssim, nonhole_ssim = np.mean(ssim_map), np.mean(ssim_map)
    else:
        hole_ssim = np.sum(ssim_map*dmask) / np.sum(dmask)
        nonhole_ssim = np.sum(ssim_map*(1-dmask)) / np.sum(1-dmask)

    return np.mean(ssim_map), hole_ssim, nonhole_ssim


if __name__ == "__main__":
    im1 = real_B
    im2 = fake_B
    Metric = ImageInpaintingMetric(im1, im2)
#     Metric.evaluate()
    Metric.printMeric()