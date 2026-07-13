from __future__ import print_function, division
import os
import cv2
import random
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings

import matplotlib.pyplot as plt

# 消除运行中的红色字体
warnings.filterwarnings("ignore")

# 定义位置挑选
def select_Rx_positions(Rx, num_Rx, interations):
    selections = []
    for _ in range(interations):
        selected_indices = np.random.choice(len(Rx), num_Rx, replace=False)
        selected_positions = Rx[selected_indices]
        selections.append(selected_positions)
    return selections

class Multi_loader(Dataset):
    def __init__(self, phase='train',
                 data="data/",
                 data_type="land",    # air or land
                 lvm_type="sam",   # sam or samhq
                 channel="False",     # True or False
                 sample_nums=20,
                 pick_times=1000,
                 ):

        self.len = 3
        self.height = 512
        self.width = 512
        self.phase = phase
        self.data = data
        self.data_type = data_type
        self.lvm_type = lvm_type
        self.channel = channel
        self.sample_nums = sample_nums

        # phase
        if self.phase == 'train':
            self.pick_times = pick_times

        elif self.phase == 'val':
            self.pick_times = 100

        else:
            self.pick_times = 10

        # path
        if self.data_type == 'air':
            if self.channel == "True":
                self.obs_path = self.data + "Air/rayleigh/"
                if self.lvm_type == "sam":
                    self.img_path = self.data + "img/rayleigh/label/"
                    self.lvm_path = self.data + "img/rayleigh/seg/label/"  # blend or label

                else:
                    self.img_path = self.data + "img/rayleigh/label/"
                    self.lvm_path = self.data + "img/rayleigh/seghq/label/"  # blend or label

            else:
                self.obs_path = self.data + "Air/init/"
                if self.lvm_type == "sam":
                    self.img_path = self.data + "img/init/label/"
                    self.lvm_path = self.data + "img/init/seg/blend/"  # blend or label

                else:
                    self.img_path = self.data + "img/init/label/"
                    self.lvm_path = self.data + "img/init/seghq/blend/"  # blend or label
        else:
            if self.channel == "True":
                self.obs_path = self.data + "Land/rayleigh/"
                if self.lvm_type == "sam":
                    self.img_path = self.data + "img/rayleigh/label/"
                    self.lvm_path = self.data + "img/rayleigh/seg/label/"  # blend or label

                else:
                    self.img_path = self.data + "img/rayleigh/label/"
                    self.lvm_path = self.data + "img/rayleigh/seghq/label/"  # blend or label

            else:
                self.obs_path = self.data + "Land/init/"
                if self.lvm_type == "sam":
                    self.img_path = self.data + "img/init/label/"
                    self.lvm_path = self.data + "img/init/seg/blend/"  # blend or label

                else:
                    self.img_path = self.data + "img/init/label/"
                    self.lvm_path = self.data + "img/init/seghq/blend/"  # blend or label

    def __len__(self):
        return self.len * self.pick_times

    def __getitem__(self, idx):

        idx = np.floor(idx/self.pick_times).astype(int)
        name = str(idx)

        # SAM
        if self.data_type == "air":
            if self.channel == "True":
                obs_path = self.obs_path + "train_test_split_" + str(idx + 21) + ".npz"
                if self.lvm_type == "sam":
                    img_path = self.img_path + "1.png"
                    lvm_path = self.lvm_path + "1.png"  # blend or label

                else:
                    img_path = self.img_path + "1.png"
                    lvm_path = self.lvm_path + "1.png"  # blend or label

            else:
                obs_path = self.obs_path + "train_test_split_" + str(idx + 21) + ".npz"
                if self.lvm_type == "sam":
                    img_path = self.img_path + "1.png"
                    lvm_path = self.lvm_path + "1.png"  # blend or label

                else:
                    img_path = self.img_path + "1.png"
                    lvm_path = self.lvm_path + "1.png"  # blend or label

        else:
            if self.channel == "True":
                obs_path = self.obs_path + "train_test_split_" + str(idx + 381) + ".npz"
                if self.lvm_type == "sam":
                    img_path = self.img_path + "1.png"
                    lvm_path = self.lvm_path + "1.png"  # blend or label

                else:
                    img_path = self.img_path + "1.png"
                    lvm_path = self.lvm_path + "1.png"  # blend or label

            else:
                obs_path = self.obs_path + "train_test_split_" + str(idx + 381) + ".npz"
                if self.lvm_type == "sam":
                    img_path = self.img_path + "1.png"
                    lvm_path = self.lvm_path + "1.png"  # blend or label

                else:
                    img_path = self.img_path + "1.png"
                    lvm_path = self.lvm_path + "1.png"  # blend or label

        # print(obs_path)
        # print(img_path)
        # print(lvm_path)

        if self.phase == 'train':
            data = np.load(obs_path)
            data = data['train']

        elif self.phase == 'val':
            data = np.load(obs_path)
            data = data['test']

        else:
            data = np.load(obs_path)
            data = data['test']

        # 掩码
        mask_all = np.where(data == 0, 0, 1)

        # sampling
        nonzero_coords = list(zip(*np.nonzero(data)))

        sample = random.sample(nonzero_coords, self.sample_nums)

        imRx = np.zeros((self.width, self.height))

        for m in range(self.sample_nums):
            single_Rx = sample[m]
            imRx[single_Rx[0], single_Rx[1]] = 1

        mask = imRx
        sample = mask * data

        # Aerial map
        img = np.asarray(io.imread(img_path))
        img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_LANCZOS4)

        # SAM map
        lvm_img = np.asarray(io.imread(lvm_path))
        lvm_img = cv2.resize(lvm_img, (self.height, self.width), interpolation=cv2.INTER_LANCZOS4)

        # lvm_img = np.where(lvm_img == 0, 0, 1)

        # target
        if self.data_type == "air":
            simulation_data = np.load(self.obs_path + "simulation_" + str(idx + 21) + ".npz")
            simulation_data = simulation_data['RSRP']
            measurement_data = np.load(self.obs_path + "measurement_" + str(idx + 21) + ".npz")
            measurement_data = measurement_data['RSRP']

        else:
            simulation_data = np.load(self.obs_path + "simulation_" + str(idx + 381) + ".npz")
            simulation_data = simulation_data['RSRP']
            measurement_data = np.load(self.obs_path + "measurement_" + str(idx + 381) + ".npz")
            measurement_data = measurement_data['RSRP']

        # target
        measurement_data[measurement_data == 0] = simulation_data[measurement_data == 0]

        # # 需要消融
        # build = (build - 1964) / 57

        # # 需要消融
        # building_path = self.data + "building_altitude_range.npz"

        # normalization
        sample[sample == 0] = -120
        sample = (sample + 120) / 70

        target = (measurement_data + 120) / 70

        # # visualization
        # plt.imshow(img, cmap='viridis', origin='lower')
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(lvm_img, cmap='viridis', origin='lower')
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(sample, cmap='viridis', origin='lower')
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(target, cmap='viridis', origin='lower')
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(mask_all, cmap='viridis', origin='lower')
        # plt.colorbar()
        # plt.show()

        # transfer dimension
        img = np.transpose(img, (2, 0, 1))
        # lvm_img = np.expand_dims(lvm_img, axis=0)
        lvm_img = np.transpose(lvm_img, (2, 0, 1))
        sample = np.expand_dims(sample, axis=0)
        mask_all = np.expand_dims(mask_all, axis=0)
        target = np.expand_dims(target, axis=0)

        # To tensor
        img = torch.from_numpy(np.asarray(img / 255, dtype=np.float32))
        lvm_img = torch.from_numpy(np.asarray(lvm_img, dtype=np.float32))
        sample = torch.from_numpy(np.asarray(sample, dtype=np.float32))
        mask_all = torch.from_numpy(np.asarray(mask_all, dtype=np.float32))
        target = torch.from_numpy(np.asarray(target, dtype=np.float32))

        return img, lvm_img, sample, mask_all, target, name

def test():
    dataset = Multi_loader(phase='test')
    loader = DataLoader(dataset, batch_size=5, shuffle=True)

    for x, y, z, m, w, name in loader:
        print(x.shape, y.shape, z.shape, m. shape, w.shape, name)

if __name__ == "__main__":
    test()

