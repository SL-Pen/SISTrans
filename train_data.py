"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: train_data.pyd
about: build the training dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize


# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir):
        super().__init__()
        train_list = train_data_dir + 'Rain13k.txt'
        with open(train_list) as f:
            contents = f.readlines()
            rain_names = [i.strip() for i in contents]
            gt_names = rain_names

        self.rain_names = rain_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size

        rain_name = self.rain_names[index]
        gt_name = self.gt_names[index]

        rain_img = Image.open(self.train_data_dir + 'input/' + rain_name).convert('RGB')

        try:
            gt_img = Image.open(self.train_data_dir + 'target/' + gt_name).convert('RGB')
        except:
            gt_img = Image.open(self.train_data_dir + 'target/' + gt_name).convert('RGB')

        width, height = rain_img.size
        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        rain_crop_img = rain_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_rain = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        rain = transform_rain(rain_crop_img)
        gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(rain.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return rain, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.rain_names)

