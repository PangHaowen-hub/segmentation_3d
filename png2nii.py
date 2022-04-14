import SimpleITK as sitk
import os
from PIL import Image
import numpy as np
from tqdm import trange


def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.png':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


if __name__ == '__main__':
    nii_path = r'G:\my_lobe_data\after\_SJ_test\RU\mask\zhangqinghua_mask.nii.gz'
    mask_path = r'F:\my_code\segmentation_3d\data\images\test\SJ\after\png\RU\RU_003_RU'
    save_path = r'F:\my_code\segmentation_3d\data\images\test\SJ\after\png\RU'
    sitk_img = sitk.ReadImage(nii_path)
    img_arr = sitk.GetArrayFromImage(sitk_img)

    img_list = get_listdir(mask_path)
    img_list.sort()
    new_mask = np.zeros_like(img_arr)
    for i in trange(len(img_list)):
        image = Image.open(img_list[i])
        new_mask[i, :, :] = image
    new_mask = sitk.GetImageFromArray(new_mask)
    new_mask.SetDirection(sitk_img.GetDirection())
    new_mask.SetSpacing(sitk_img.GetSpacing())
    new_mask.SetOrigin(sitk_img.GetOrigin())
    _, fullflname = os.path.split(nii_path)
    sitk.WriteImage(new_mask, os.path.join(save_path, fullflname))
