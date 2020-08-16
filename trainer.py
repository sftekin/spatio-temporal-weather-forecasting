import time
import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from models.adaptive_normalizer import AdaptiveNormalizer


class Trainer:
    def __init__(self, model, criterion, optimizer, clip):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.clip = clip
        self.normalizer = AdaptiveNormalizer()

    def step(self, input_tensor, output_tensor):
        self.optimizer.zero_grad()
        pred = self.model(input_tensor)
        loss = self.criterion(output_tensor, pred)

        if self.model.training:
            loss.backward()
            nn.utils.clip_grad_norm(self.model.paremeters(), self.clip)
            self.optimizer.step()

        return loss


def train(model, batch_generator):
    """

    :param model:
    :param batch_generator:
    :return:
    """
    train_loss = []
    val_loss = []

    tolerance = 0
    best_epoch = 0
    best_val_loss = 1e6
    evaluation_val_loss = best_val_loss
    best_dict = model.state_dict()

    optimizer = optim.Adam(model.parameters(),
                           lr=self.learning_rate,
                           weight_decay=self.l2_reg)

    for epoch in range(self.num_epochs):
        # train and validation loop
        start_time = time.time()

        epoch_time = time.time() - start_time
