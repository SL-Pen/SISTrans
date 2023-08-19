"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: utils.py
about: all utilities
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage.metrics import structural_similarity as compare_ssim


def to_psnr(derain, gt):
    mse = F.mse_loss(derain, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(derain, gt):
    derain_list = torch.split(derain, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    derain_list_np = [derain_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                      range(len(derain_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(derain_list))]
    ssim_list = [compare_ssim(derain_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in
                 range(len(derain_list))]

    return ssim_list


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img


def norm_range(t, range):
    if range is not None:
        return norm_ip(t, range[0], range[1])
    else:
        return norm_ip(t, t.min(), t.max())


def validation(net, val_data_loader, device, category, save_tag=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            rain, gt, image_name = val_data
            rain = rain.to(device)
            gt = gt.to(device)
            derain = net(rain)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(derain, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(derain, gt))

        # --- Save image --- #
        if save_tag:
            save_image(derain, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(derain, image_name, category):
    derain_images = torch.split(derain, 1, dim=0)
    batch_num = len(derain_images)

    for ind in range(batch_num):
        utils.save_image(derain_images[ind], './{}_results/{}'.format(category, image_name[ind][:-3] + 'png'))


def print_log(log_file, epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # --- Write the training log --- #
    with open(log_file, 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), one_epoch_time, epoch, num_epochs, train_psnr,
                val_psnr, val_ssim), file=f)


