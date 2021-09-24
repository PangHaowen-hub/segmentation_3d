import torch
import argparse

from tqdm import trange
from unet.unet_model import UNet
from dataset_2d import test_dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if file[-4:] == '0000':
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=3).to(device)  # TODO:改类别数
    color_map = get_color_map_list(256)
    model.eval()
    model.load_state_dict(torch.load('./UNet_RML20.pth', map_location='cuda'))

    dir = "./data/images/test/after/RM"
    files_path = get_listdir(dir)
    for files in trange(15, len(files_path)):
        dataset = test_dataset(rootpth=files_path[files])
        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        save_path = files_path[files][:-5] + '_right'
        os.mkdir(save_path)
        for x, name in test_dataloader:
            inputs = x.to(device)
            outputs = model(inputs)
            out = outputs.argmax(dim=1).squeeze().detach().cpu().numpy()
            mask_pil = Image.fromarray(out.astype(np.uint8), mode='P')
            mask_pil.putpalette(color_map)
            mask_pil.save(os.path.join(save_path, name[0]))
