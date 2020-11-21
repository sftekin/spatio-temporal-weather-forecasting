import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from copy import deepcopy
from models.adaptive_normalizer import AdaptiveNormalizer
from models.weather_model import WeatherModel


def train(batch_generator, trainer_conf, model_conf):
    """

    :param model:
    :param batch_generator:
    :param trainer_conf:
    :param model_conf:
    :return:
    """
    device = torch.device(trainer_conf['device'])

    model = WeatherModel(device=device, **model_conf).to(device)
    model.train()

    normalizer = AdaptiveNormalizer()

    optimizer = optim.Adam(model.parameters(),
                           lr=trainer_conf['learning_rate'])
    criterion = nn.MSELoss()

    batch_size = batch_generator.dataset_params['batch_size']
    train_loss = []
    val_loss = []
    for epoch in range(trainer_conf['num_epochs']):
        # train and validation loop
        start_time = time.time()

        running_loss = 0
        for idx, (x, y, f_x, f_y) in enumerate(batch_generator.generate(dataset_name='train')):
            print('\rtrain:{}/{}, loss:{:.2f}'.format(idx, batch_generator.num_iter('train'),
                                                      running_loss), flush=True, end='')

            x, y, f_x, f_y = [prep_input(i, device) for i in [x, y, f_x, f_y]]
            hidden = model.init_hidden(batch_size)

            pred = model.forward(x, y, f_x, f_y, hidden)

            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), trainer_conf['clip'])

            # take step in classifier's optimizer
            optimizer.step()

            torch.cuda.empty_cache()
            running_loss += loss.mean().detach().cpu().numpy()

        epoch_time = time.time() - start_time
        print('Training Loss: {:.2f}'.format(running_loss))


def prep_input(x, device):
    x = x.float().to(device)
    # (b, t, m, n, d) -> (b, t, d, m, n)
    x = x.permute(0, 1, 4, 2, 3)
    return x

