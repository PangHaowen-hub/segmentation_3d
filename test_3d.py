import torch
import argparse
from vnet3d import VNet
from unet3d import UNet3D
from torch import optim
from dataset_3d import test_dataset
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import os
import tqdm
import torchio
from torchio.transforms import ZNormalization, CropOrPad, Compose
import SimpleITK as sitk

def test(model):
    model.eval()
    model.load_state_dict(torch.load(args.load, map_location='cuda'))
    batch_size = args.batch_size
    source_test_dir = r'./data_3d/img'
    dataset = test_dataset(source_test_dir)
    znorm = ZNormalization()
    patch_overlap = 64, 64, 64
    patch_size = 128

    for i, subj in enumerate(dataset.subjects):
        subj = znorm(subj)
        grid_sampler = torchio.inference.GridSampler(subj, patch_size, patch_overlap)
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        with torch.no_grad():
            for patches_batch in tqdm.tqdm(patch_loader):
                input_tensor = patches_batch['source'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]
                outputs = model(input_tensor)  # outputs torch.Size([1, 6, 128, 128, 128])
                aggregator.add_batch(outputs, locations)
        output_tensor = aggregator.get_output_tensor()
        output_tensor = output_tensor.argmax(dim=0).unsqueeze(0)
        affine = subj['source']['affine']
        output_image = torchio.ScalarImage(tensor=output_tensor.numpy().astype(np.uint8), affine=affine)
        output_image.save('temp.nii.gz')
        print('保存成功')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VNet(elu=True, in_channels=1, classes=6).to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--load', dest='load', type=str, default='VNet_80.pth', help='the path of the .pth file')
    args = parser.parse_args()
    test(model)
