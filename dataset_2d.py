import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json


def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.png':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


class MyDataset(Dataset):
    def __init__(self, rootpth, mode='train', *args, **kwargs):
        super(MyDataset, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        with open('label_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        # parse img directory
        self.imgs = {}
        impth = osp.join(rootpth, 'images', mode)
        folders = os.listdir(impth)
        impths = [osp.join(impth, el) for el in folders]
        folders_name = [el.replace('.png', '') for el in folders]
        imgnames = folders_name
        self.imgs.update(dict(zip(folders_name, impths)))

        # parse gt directory
        self.labels = {}
        gtpth = osp.join(rootpth, 'masks', mode)
        folders = os.listdir(gtpth)
        lbpths = [osp.join(gtpth, el) for el in folders]
        folders_name = [el.replace('.png', '') for el in folders]
        self.labels.update(dict(zip(folders_name, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)

        # pre-processing
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        fn = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        label = np.squeeze(label)
        return img, label, fn

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label


class test_dataset(Dataset):
    def __init__(self, rootpth, mode='test', *args, **kwargs):
        super(test_dataset, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.imgs = {}
        impth = osp.join(rootpth, 'images', mode, 'LL')  # TODO：需要改此处
        self.img_list = get_listdir(impth)
        self.len = len(self.img_list)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        impth = self.img_list[idx]
        img = Image.open(impth)
        img = self.to_tensor(img)
        _, fullflname = os.path.split(impth)
        return img, fullflname

    def __len__(self):
        return self.len
