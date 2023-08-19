import time
import torch
import argparse
from torch.utils.data import DataLoader
from Test_data import TestData
from model import SISTrans
from utils import validation
import os

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters')
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-test_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (derain)', default='derain', type=str)
args = parser.parse_args()

test_batch_size = args.test_batch_size
category = args.category

print('--- Hyper-parameters for testing ---')
print(
    'test_batch_size: {}\ncategory: {}'
    .format(test_batch_size, category))

# --- Set category-specific hyper-parameters  --- #
test_data_dir = './data/test/Test100/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
test_data_loader = DataLoader(TestData(test_data_dir), batch_size=test_batch_size, shuffle=False, num_workers=0)
# TestData:return rain, gt, rain_name

net = SISTrans()

# --- Multi-GPU --- #
net = net.to(device)

nec_loc = "./model_weights/"
# --- Load the network weight --- #
net.load_state_dict(torch.load(os.path.join(nec_loc, 'derain_Derain_best_111_35.pth'), map_location=device))

# --- Use the evaluation model in testing --- #
# net.eval()
print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim = validation(net, test_data_loader, device, category, save_tag=False)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))
