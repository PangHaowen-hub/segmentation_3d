import torch
from torchio.transforms import CropOrPad
import os
from tqdm import trange
import SimpleITK as sitk
import numpy as np


def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


if __name__ == '__main__':
    img_path = r'F:\my_lobe_data\before\all_lobe_512\imgs_rename'
    mask_path = r'F:\my_lobe_data\before\all_lobe_512\masks_rename'
    img_list = get_listdir(img_path)
    img_list.sort()
    mask_list = get_listdir(mask_path)
    mask_list.sort()
    transform = CropOrPad((256, 512, 512), padding_mode='reflect')
    for i in trange(len(img_list)):
        sitk_img = sitk.ReadImage(img_list[i])
        img_arr = sitk.GetArrayFromImage(sitk_img)
        img_arr = torch.from_numpy(np.expand_dims(img_arr, 0))
        img = transform(img_arr)
        out = img.squeeze().detach().cpu().numpy()
        new_img = sitk.GetImageFromArray(out)
        new_img.SetDirection(sitk_img.GetDirection())
        new_img.SetSpacing(sitk_img.GetSpacing())
        new_img.SetOrigin(sitk_img.GetOrigin())
        _, fullflname = os.path.split(img_list[i])
        sitk.WriteImage(new_img, os.path.join(r'D:\my_code\u_net_multiple_classification\data_3d\img', fullflname))

        sitk_mask = sitk.ReadImage(mask_list[i])
        mask_arr = sitk.GetArrayFromImage(sitk_mask).astype(np.uint8)
        mask_arr = torch.from_numpy(np.expand_dims(mask_arr, 0))
        mask = transform(mask_arr)
        out_mask = mask.squeeze().detach().cpu().numpy()
        new_mask = sitk.GetImageFromArray(out_mask)
        new_mask.SetDirection(sitk_mask.GetDirection())
        new_mask.SetSpacing(sitk_mask.GetSpacing())
        new_mask.SetOrigin(sitk_mask.GetOrigin())
        _, fullflname = os.path.split(mask_list[i])
        sitk.WriteImage(new_mask, os.path.join(r'D:\my_code\u_net_multiple_classification\data_3d\mask', fullflname))
