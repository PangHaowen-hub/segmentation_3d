import torch
import argparse
from vnet3d import VNet
from unet3d import UNet3D
from dataset_3d import test_dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import tqdm
import torchio
from torchio.transforms import ZNormalization, CropOrPad, Compose, Resample, Resize
import SimpleITK as sitk


def test(model):
    model.eval()
    model.load_state_dict(torch.load(args.load, map_location='cuda'))
    batch_size = args.batch_size
    source_test_dir = r'F:\my_code\segmentation_3d\data_SJ\after\img\RL'
    save_path = r'F:\my_code\segmentation_3d\data_SJ\after\pred_3d\RL\3dUNet\right'
    dataset = test_dataset(source_test_dir)
    patch_overlap = 64, 64, 64
    patch_size = 128

    for i, subj in enumerate(dataset.test_set):
        grid_sampler = torchio.inference.GridSampler(subj, patch_size, patch_overlap)  # 从图像中提取patch
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size)
        aggregator = torchio.inference.GridAggregator(grid_sampler, 'average')  # 用于聚合patch推理结果
        with torch.no_grad():
            for patches_batch in tqdm.tqdm(patch_loader):
                input_tensor = patches_batch['source'][torchio.DATA].to(device).float()
                outputs = model(input_tensor)  # outputs torch.Size([1, 6, 128, 128, 128])
                locations = patches_batch[torchio.LOCATION]  # patch的位置信息
                aggregator.add_batch(outputs, locations)
        output_tensor = aggregator.get_output_tensor()  # 获取聚合后volume
        affine = subj['source']['affine']
        output_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
        out_transform = Resize(dataset.get_shape(i)[1:])  # 恢复原始大小
        output_image = out_transform(output_image)

        name = subj['source']['path']
        _, fullflname = os.path.split(name)
        new_mask = sitk.GetImageFromArray(output_image.data.argmax(dim=0).int().permute(2, 1, 0))
        new_mask.SetSpacing(output_image.spacing)
        new_mask.SetDirection(output_image.direction)
        new_mask.SetOrigin(output_image.origin)
        sitk.WriteImage(new_mask, os.path.join(save_path, fullflname))
        # output_image.save(os.path.join(save_path, fullflname))
        # print(os.path.join(save_path, fullflname) + '保存成功')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = VNet(elu=True, in_channels=1, classes=3).to(device)
    model = UNet3D(in_channels=1, out_channels=3).to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--load', dest='load', type=str, default='./UNet3d_RLL/UNet3d_RLL_99.pth',
                        help='the path of the .pth file')
    args = parser.parse_args()
    test(model)
