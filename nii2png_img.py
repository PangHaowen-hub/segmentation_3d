import SimpleITK as sitk
import os
from PIL import Image
import numpy as np
from tqdm import trange


def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def nii2png(img, save_path):
    sitk_img = sitk.ReadImage(img)
    img_arr = sitk.GetArrayFromImage(sitk_img)
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    img_arr[img_arr > MAX_BOUND] = MAX_BOUND
    img_arr[img_arr < MIN_BOUND] = MIN_BOUND
    img_arr = (img_arr - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) * 255
    _, fullflname = os.path.split(img)
    for i in trange(img_arr.shape[0]):
        temp = img_arr[i, :, :].astype(np.uint8)
        img_pil = Image.fromarray(temp)
        img_pil.save(os.path.join(save_path, 'images', 'test', 'LL', fullflname + '_' + str(i) + '.png'))


if __name__ == '__main__':
    img_path = r'F:\my_lobe_data\after\LL\imgs_rename'
    save_path = r'D:\my_code\u_net_multiple_classification\data'
    img_list = get_listdir(img_path)
    img_list.sort()
    for i in trange(len(img_list)):
        nii2png(img_list[i], save_path)
