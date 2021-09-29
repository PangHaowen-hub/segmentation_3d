import torch
import argparse
from vnet3d import VNet
from unet3d import UNet3D
from torch import optim
from dataset_3d import MyDataset, test_dataset
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import os
import torchio
from torchio.transforms import ZNormalization, CropOrPad, Compose, Resample, Resize
import tqdm
import SimpleITK as sitk
from dice_3d import dice_3d, get_listdir

respth = './res'
if not os.path.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def setup_logger(logpth):
    logfile = 'VNet-RML-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = os.path.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def train(model):
    model.train()
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location='cuda'))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    source_train_dir = r'./data_3d/model_RML/img'
    label_train_dir = r'./data_3d/model_RML/mask'
    train_dataset = MyDataset(source_train_dir, label_train_dir)
    train_dataloader = DataLoader(train_dataset.queue_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)

    for epoch in range(args.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, args.num_epochs))
        logger.info('-' * 10)
        dataset_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for i, batch in enumerate(train_dataloader):
            x = batch['source']['data']
            y = batch['label']['data']
            x = x.cuda()
            y = torch.squeeze(y, 1).long()
            y = y.cuda()

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            logger.info("%d/%d,train_loss:%0.5f" % (step, dataset_size // train_dataloader.batch_size, loss.item()))
        logger.info("epoch %d loss:%0.5f" % (epoch, epoch_loss))
        torch.save(model.state_dict(), './VNet_RML/VNet_RML_%d.pth' % epoch)
        if epoch % 10 == 0:
            source_test_dir = r'./data_3d/test/RM/img'
            save_path = r'./data_3d/test/RM/pred_right'
            dataset = test_dataset(source_test_dir)
            patch_overlap = 64, 64, 64
            patch_size = 128
            for i, subj in enumerate(dataset.test_set):
                grid_sampler = torchio.inference.GridSampler(subj, patch_size, patch_overlap)  # 从图像中提取patch
                patch_loader = torch.utils.data.DataLoader(grid_sampler, 1)
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
                out_transform = Resize(dataset.get_shape(i)[1:])
                output_image = out_transform(output_image)

                name = subj['source']['path']
                _, fullflname = os.path.split(name)
                new_mask = sitk.GetImageFromArray(output_image.data.argmax(dim=0).int().permute(2, 1, 0))
                new_mask.SetSpacing(output_image.spacing)
                new_mask.SetDirection(output_image.direction)
                new_mask.SetOrigin(output_image.origin)
                sitk.WriteImage(new_mask, os.path.join(save_path, fullflname))

            mask_path = r'./data_3d/mask/RM'
            pred_path = r'./data_3d/test/RM/pred_right'
            mask = get_listdir(mask_path)
            mask.sort()
            pred = get_listdir(pred_path)
            pred.sort()
            dice = 0
            for i in range(len(mask)):
                dice += dice_3d(mask[i], pred[i], 1)
            logger.info(dice / len(mask))
            dice = 0
            for i in range(len(mask)):
                dice += dice_3d(mask[i], pred[i], 2)
            logger.info(dice / len(mask))


if __name__ == '__main__':
    setup_logger(respth)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VNet(elu=True, in_channels=1, classes=3).to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--load', dest='load', type=str, help='the path of the .pth file')
    parser.add_argument('--epoch', dest='num_epochs', type=int, default=100, help='num_epochs')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0005, help='learning_rate')
    args = parser.parse_args()
    train(model)
