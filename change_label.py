import os
import numpy as np
from PIL import Image
import copy
import tqdm

def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.png':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def get_color_map_list(num_classes):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


path = r'D:\my_code\segmentation_3d\data\masks\train\after\RUL'
name_list = get_listdir(path)
color_map = get_color_map_list(256)
for i in tqdm.tqdm(name_list):
    image = Image.open(i)
    image = np.array(image)
    temp = copy.deepcopy(image)
    image[temp > 3] = 0
    image[temp == 2] = 1
    image[temp == 3] = 2
    lbl_pil = Image.fromarray(image.astype(np.uint8), mode='P')
    lbl_pil.putpalette(color_map)
    lbl_pil.save(i)
