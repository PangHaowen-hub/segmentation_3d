import torch
import argparse
from vnet3d import VNet
from unet3d import UNet3D
from torch import optim
from dataset_3d import MyDataset
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import os

respth = './res'
if not os.path.exists(respth):
    os.makedirs(respth)
logger = logging.getLogger()


def setup_logger(logpth):
    logfile = 'VNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = os.path.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def train(model):
    model.train()
    # model.load_state_dict(torch.load(args.load, map_location='cuda'))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    source_train_dir = r'./data_3d/img'
    label_train_dir = r'./data_3d/mask'
    train_dataset = MyDataset(source_train_dir, label_train_dir)
    train_dataloader = DataLoader(train_dataset.queue_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)

    for epoch in range(args.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        logger.info('-' * 10)
        dataset_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for i, batch in enumerate(train_dataloader):
            x = batch['source']['data']
            y = batch['label']['data']
            x = x.type(torch.FloatTensor).cuda()
            y = np.squeeze(y, 1)
            y = y.type(torch.FloatTensor).cuda()

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            logger.info("%d/%d,train_loss:%0.5f" % (step, dataset_size // train_dataloader.batch_size, loss.item()))
        logger.info("epoch %d loss:%0.5f" % (epoch, epoch_loss))
        torch.save(model.state_dict(), 'VNet_%d.pth' % epoch)


if __name__ == '__main__':
    setup_logger(respth)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VNet(elu=True, in_channels=1, classes=6).to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--load', dest='load', type=str, default='VNet_3.pth', help='the path of the .pth file')
    parser.add_argument('--epoch', dest='num_epochs', type=int, default=100, help='num_epochs')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3, help='learning_rate')
    args = parser.parse_args()
    train(model)
