import time
import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from models.adaptive_normalizer import AdaptiveNormalizer
from models.weather_model import WeatherModel


def train(model, batch_generator, trainer_conf):
    """

    :param model:
    :param batch_generator:
    :param trainer_conf:
    :return:
    """
    device = torch.device(trainer_conf['device'])

    model = WeatherModel(**trainer_conf).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(),
                           lr=trainer_conf['learning_rate'],
                           weight_decay=trainer_conf['l2_reg'])
    criterion = nn.MSELoss()

    train_loss = []
    val_loss = []
    for epoch in range(trainer_conf['num_epochs']):
        # train and validation loop
        start_time = time.time()

        running_loss = 0
        for idx, (x, y) in enumerate(batch_generator.generate(dataset_name='train')):
            print('\rtrain:{}/{}'.format(idx, batch_generator.num_iter), flush=True, end='')
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()


        epoch_time = time.time() - start_time
