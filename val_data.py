"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: val_data.py
about: build the validation/test dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from random import randrange


# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'Test100.txt'
        with open(val_list) as f:
            contents = f.readlines()
            rain_names = [i.strip() for i in contents]
            gt_names = rain_names

        self.rain_names = rain_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]

        rain_img = Image.open(self.val_data_dir + 'input/' + rain_name).convert('RGB')
        gt_img = Image.open(self.val_data_dir + 'target/' + gt_name).convert('RGB')

        width, height = rain_img.size
        #
        wd_new = 128
        ht_new = 128

        x, y = randrange(0, width - wd_new + 1), randrange(0, height - ht_new + 1)
        rain_img = rain_img.crop((x, y, x + wd_new, y + ht_new))
        gt_img = gt_img.crop((x, y, x + wd_new, y + ht_new))

        # --- Transform to tensor --- #
        transform_rain = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        rain = transform_rain(rain_img)
        gt = transform_gt(gt_img)

        return rain, gt, rain_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)
