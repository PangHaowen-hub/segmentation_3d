import torch.utils.data as data
from torchvision.transforms import transforms
import numpy as np
import logging
import os
import SimpleITK as sitk
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torchio.transforms import ZNormalization, CropOrPad, Compose
import torchio as tio


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


class MyDataset(data.Dataset):
    def __init__(self, imgs_path, mask_path):
        self.img_list = get_listdir(imgs_path)
        self.img_list.sort()
        self.mask_list = get_listdir(mask_path)
        self.mask_list.sort()
        self.subjects = []
        for (image_path, label_path) in zip(self.img_list, self.mask_list):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)
        self.transforms = self.transform()
        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)
        queue_length = 5
        samples_per_volume = 5
        patch_size = 128
        self.queue_dataset = Queue(self.training_set, queue_length, samples_per_volume, UniformSampler(patch_size))

    def transform(self):
        crop_or_pad_size = (512, 512, 320)
        training_transform = Compose([
            CropOrPad(crop_or_pad_size, padding_mode='reflect'),
            ZNormalization(),
        ])
        return training_transform

class test_dataset(data.Dataset):
    def __init__(self, imgs_path):
        self.img_list = get_listdir(imgs_path)
        self.img_list.sort()
        self.subjects = []
        for image_path in self.img_list:
            subject = tio.Subject(
                source=tio.ScalarImage(image_path)
            )
            self.subjects.append(subject)

        self.training_set = tio.SubjectsDataset(self.subjects, transform=None)


