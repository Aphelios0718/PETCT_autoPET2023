import os
import glob
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
from email.header import make_header
from monai.config import print_config

from models.colean_all import CoLearnUNet
from datasets.seg_datasets import PETCT_seg

from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric, compute_meandice

from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch, DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = CoLearnUNet(n_seg_classes=2)
        
        self.loss_function = DiceFocalLoss(to_onehot_y=True,softmax=True,include_background=False,batch=True)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
        self.post_label = AsDiscrete(to_onehot=True, n_classes=2)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.args = parse_opts()

    def forward(self, x):
        return self._model(x)

    def prepare_data(self, data_dir):
        petct_npzs = glob.glob(f"{data_dir}\petct_arr.npz")
        self.val_ds = PETCT_infer(data_list=petct_npzs)

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=0)
        return val_loader


def segment_PETCT(ckpt_path, data_dir, export_dir):
    print("starting")
    
    device = torch.device("cuda:0")
    roi_size = (128, 128, 128)
    sw_batch_size = 4
    
    net = torch.load(ckpt_path)
    net.eval()
    net.to(device)
    net.prepare_data(data_dir)
    
    with torch.no_grad():
        for i, val_data in enumerate(net.val_dataloader()):
            mask_out = sliding_window_inference(
                val_data["petct"].to(device), 
                roi_size, 
                sw_batch_size, 
                net)
            
            mask_out = torch.argmax(mask_out, dim=1).detach().cpu().numpy().squeeze()
            mask_out = mask_out.astype(np.uint8)               
            print("done inference")
            
            # needs to be loaded to recover nifti header and export mask
            PT = nib.load(os.path.join(data_dir,"SUV.nii.gz"))  
            pet_affine = PT.affine
            PT = PT.get_fdata()
            
            mask_export = nib.Nifti1Image(mask_out, pet_affine)
            print(os.path.join(export_dir, "PRED.nii.gz"))
            nib.save(mask_export, os.path.join(export_dir, "PRED.nii.gz"))
            print("done writing")


def run_inference(ckpt_path='/opt/algorithm/epoch=777-step=64573.ckpt', 
                  data_dir='/opt/algorithm/', 
                  export_dir='/output/images/automated-petct-lesion-segmentation/'
                  ):
    
    segment_PETCT(ckpt_path, data_dir, export_dir)


if __name__ == '__main__':
    run_inference()