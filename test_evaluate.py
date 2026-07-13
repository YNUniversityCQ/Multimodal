import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from lib1 import loaders, modules
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader

from lib1.dcnunet import SimpleUNet

def masked_ssim(img1, img2, valid_mask):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if not np.any(valid_mask):
        return 0.0

    # 用有效区域均值填充无效区域，避免无效区影响局部统计
    m1 = img1[valid_mask].mean()
    m2 = img2[valid_mask].mean()

    img1_f = img1.copy()
    img2_f = img2.copy()
    img1_f[~valid_mask] = m1
    img2_f[~valid_mask] = m2

    sigma = 1.5
    C1 = (0.01 * (-50)) ** 2
    C2 = (0.03 * (-50)) ** 2

    mu1 = gaussian_filter(img1_f, sigma)
    mu2 = gaussian_filter(img2_f, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1_f * img1_f, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2_f * img2_f, sigma) - mu2_sq
    sigma12 = gaussian_filter(img1_f * img2_f, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 只在有效区域求平均
    return float(ssim_map[valid_mask].mean())


# loading model
torch.cuda.set_device(0)
device = torch.device("cuda")

# model = modules.Multi_model()
# model.load_state_dict(torch.load('model_result/main_model.pt'))
# model.to(device)

model = SimpleUNet(in_ch=4, out_ch=1, mode='unet', channel=32, depth=5, img_size=512)
model.load_state_dict(torch.load('model_result/main_model.pt'))
model.cuda()

def main_worker():

    # loading test data
    test_data = loaders.Multi_loader(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)

    interation = 0
    err1 = []
    err2 = []
    err3 = []
    err4 = []
    err5 = []

    for img, lvm_img, sample, mask, target, img_name in tqdm(test_dataloader):
        interation += 1

        img, lvm_img, sample = img.cuda(), lvm_img.cuda(), sample.cuda()
        mask, target = mask.cuda(), target.cuda()

        with torch.no_grad():
            pre = model(img, lvm_img, sample)

            # # 可视化
            # vis = pre[0, 0].detach().cpu().numpy()
            #
            # plt.figure()
            # plt.imshow(vis, cmap='viridis')
            # plt.colorbar()
            # plt.title("Prediction")
            # plt.axis("off")
            # plt.show()

            # 掩码覆盖
            pre = pre * mask
            target = target * mask

        # 取有效点位置
        valid = mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(bool)

        # target
        test1 = torch.tensor([item.cpu().detach().numpy() for item in target]).cuda()
        test1 = test1.squeeze(0)
        test1 = test1.squeeze(0)
        im = test1.cpu().numpy()
        im = (im * 70) - 120

        # predict
        test = torch.tensor([item.cpu().detach().numpy() for item in pre]).cuda()
        test = test.squeeze(0)
        test = test.squeeze(0)
        im1 = test.cpu().numpy()
        im1 = (im1 * 70) - 120

        im_v = im[valid]
        im1_v = im1[valid]

        # calculate rmse (valid only)
        rmse1 = np.sqrt(np.mean((im_v - im1_v) ** 2))
        err1.append(rmse1)

        # calculate nmse (valid only)
        nmse1 = np.mean((im_v - im1_v) ** 2) / np.mean((0 - im_v) ** 2)
        err2.append(nmse1)

        # calculate mae (valid only)
        mae1 = np.mean(np.abs(im_v - im1_v))
        err3.append(mae1)

        # calculate psnr (valid only)
        mse = np.mean((im_v - im1_v) ** 2)
        psnr1 = 10 * np.log10(((-50) ** 2) / mse)
        err4.append(psnr1)

        # calculate ssim
        ssim1 = masked_ssim(im, im1, valid)
        err5.append(ssim1)

    rmse_err = sum(err1)/len(err1)
    nmse_err = sum(err2) / len(err2)
    mae_err = sum(err3) / len(err3)
    psnr_err = sum(err4) / len(err4)
    ssim_err = sum(err5) / len(err5)

    print('测试集均方根误差：', rmse_err)
    print('测试集归一化均方误差：', nmse_err)
    print('测试集平均绝对误差：', mae_err)
    print('测试集信噪比峰值：', psnr_err)
    print('测试集结构相似度：', ssim_err)

if __name__ == '__main__':
 main_worker()