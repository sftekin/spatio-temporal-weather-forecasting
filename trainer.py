import time
import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from models.normalizer import Normalizer


class Trainer:
    def __init__(self, num_epochs, early_stop_tolerance, norm_method,
                 loss_type, learning_rate, l2_reg, clip, device):
        """


        :param num_epochs:
        :param early_stop_tolerance:
        :param norm_method:
        :param loss_type:
        :param learning_rate:
        :param l2_reg:
        :param clip:
        :param device:
        """
        self.num_epochs = num_epochs
        self.early_stop_tolerance = early_stop_tolerance
        self.norm_method = norm_method
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.clip = clip
        self.device = device

        self.input_normalizer = Normalizer(self.norm_method)
        self.output_normalizer = Normalizer(self.norm_method)
        self.loss_dispatcher = {"l2": nn.MSELoss}

    def fit(self, model, batch_generator):
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

        data_list = []
        label_list = []
        for x, y in batch_generator.generate('train'):
            data_list.append(x.reshape(-1, *x.shape[2:]))
            label_list.append(y.reshape(-1, *y.shape[2:]))

        self.input_normalizer.fit(torch.cat(data_list))
        self.output_normalizer.fit(torch.cat(label_list))

        optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.l2_reg)

        for epoch in range(self.num_epochs):
            # train and validation loop
            start_time = time.time()

            epoch_time = time.time() - start_time
