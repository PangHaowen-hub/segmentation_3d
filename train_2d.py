import torch
import argparse
from unet.unet_model import UNet
from torch import optim
from dataset_2d import MyDataset
from dataset_2d import test_dataset
from torch.utils.data import DataLoader

import os
import logging
import time

respth = './res'
if not os.path.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def setup_logger(logpth):
    logfile = 'UNet_RLL-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))  # TODO:改log名
    logfile = os.path.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


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


def train(model):
    model.train()
    # model.load_state_dict(torch.load(args.load, map_location='cuda'))
    batch_size = args.batch_size
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataset = MyDataset(rootpth="./data", mode='train/after/RLL')  # TODO：改数据地址
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    for epoch in range(args.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        logger.info('-' * 10)
        dataset_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y, fn in train_dataloader:
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            logger.info("%d/%d,train_loss:%0.5f" % (step, dataset_size // train_dataloader.batch_size, loss.item()))
        logger.info("epoch %d loss:%0.5f" % (epoch, epoch_loss))
        torch.save(model.state_dict(), 'UNet_RLL%d.pth' % epoch)


if __name__ == '__main__':
    setup_logger(respth)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=3).to(device)  # TODO:改类别数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--load', dest='load', type=str, help='the path of the .pth file')
    parser.add_argument('--epoch', dest='num_epochs', type=int, default=100, help='num_epochs')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3, help='learning_rate')
    args = parser.parse_args()
    train(model)
