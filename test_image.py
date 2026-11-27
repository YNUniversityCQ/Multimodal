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
from collections import defaultdict
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader

# loading model
torch.cuda.set_device(0)
device = torch.device("cuda:0")
model = modules.Multi_model()
model.load_state_dict(torch.load('model_result/main_model.pt'))
model.to(device)

def main_worker():

    # loading test data
    test_data = loaders.Multi_loader(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)
    interation = 0

    for img, lvm_img, sample, mask, target, img_name in tqdm(test_dataloader):
        interation += 1

        img, lvm_img, sample = img.cuda(), lvm_img.cuda(), sample.cuda()
        mask, target = mask.cuda(), target.cuda()

        with torch.no_grad():
            pre = model(img, lvm_img, sample)
            print(interation)

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

        # name
        image_name = os.path.basename(img_name[0]).split('.')[0]

        # save and visualization
        plt.imshow(im, cmap='viridis', origin='lower')
        plt.colorbar()
        save_path = os.path.join("image_result", f'{image_name}_tar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        plt.show()

        plt.imshow(im1, cmap='viridis', origin='lower')
        plt.colorbar()
        save_path = os.path.join("image_result", f'{image_name}_pre.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        plt.show()

if __name__ == '__main__':
 main_worker()