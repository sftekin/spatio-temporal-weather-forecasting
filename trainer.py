import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class Trainer:

    def __init__(self, batch_generator, num_epochs, early_stop_tolerance, clip, learning_rate, device):
        self.batch_generator = batch_generator
        self.num_epochs = num_epochs
        self.clip = clip
        self.learning_rate = learning_rate
        self.tolerance = early_stop_tolerance
        self.device = torch.device(device)

        self.normalizer = batch_generator.normalizer
        self.criterion = nn.MSELoss()

    def fit(self, model):
        model = model.to(self.device)
        model.train()

        optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate)

        train_loss = []
        val_loss = []
        tolerance = 0
        best_val_loss = 1e6
        best_epoch = 0
        evaluation_val_loss = best_val_loss
        best_dict = model.state_dict()
        for epoch in range(self.num_epochs):
            # train and validation loop
            start_time = time.time()

            # train
            running_train_loss = self.step_loop(model=model,
                                                mode='train',
                                                optimizer=optimizer)

            # validation
            running_val_loss = self.step_loop(model=model,
                                              mode='val',
                                              optimizer=None)

            epoch_time = time.time() - start_time

            message_str = "\nEpoch: {}, Train_loss: {:.5f}, Validation_loss: {:.5f}, Took {:.3f} seconds."
            print(message_str.format(epoch + 1, running_train_loss, running_val_loss, epoch_time))

            # save the losses
            train_loss.append(running_train_loss)
            val_loss.append(running_val_loss)

            if running_val_loss < best_val_loss:
                best_epoch = epoch + 1
                best_val_loss = running_val_loss
                best_dict = deepcopy(model.state_dict())
                tolerance = 0
            else:
                tolerance += 1

            if tolerance > self.tolerance or epoch == self.num_epochs - 1:
                model.load_state_dict(best_dict)

                evaluation_val_loss = self.step_loop(model=model,
                                                     mode='val',
                                                     optimizer=None)

                message_str = "Early exiting from epoch: {}, Validation error: {:.5f}."
                print(message_str.format(best_epoch, evaluation_val_loss))
                break

            torch.cuda.empty_cache()

        print('Training finished')
        return train_loss, val_loss, evaluation_val_loss

    def step_loop(self, model, mode, optimizer):
        running_loss = 0
        batch_size = self.batch_generator.dataset_params['batch_size']
        step_fun = self.__getattribute__(mode + '_step')
        idx = 0
        for idx, (x, y, f_x, f_y) in enumerate(self.batch_generator.generate(mode)):
            print('\r{}:{}/{}'.format(mode, idx, self.batch_generator.num_iter(mode)),
                  flush=True, end='')

            x, y, f_x, f_y = [self.prep_input(i) for i in [x, y, f_x, f_y]]
            hidden = model.init_hidden(batch_size)

            loss = step_fun(model=model,
                            inputs=[x, y, f_x, f_y, hidden],
                            optimizer=optimizer)

            running_loss += loss
        running_loss /= (idx + 1)

        return running_loss

    def train_step(self, model, inputs, optimizer):
        x, y, f_x, f_y, hidden = inputs
        optimizer.zero_grad()
        pred = model.forward(x, f_x, hidden)
        loss = self.criterion(pred, y)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), self.clip)

        # take step in classifier's optimizer
        optimizer.step()

        return loss.detach().cpu().numpy()

    def val_step(self, model, inputs, optimizer):
        x, y, f_x, f_y, hidden = inputs
        pred = model.forward(x, f_x, hidden)
        if self.normalizer:
            pred = self.normalizer.inv_norm(pred, self.device)
            y = self.normalizer.inv_norm(y, self.device)

        loss = self.criterion(pred, y)

        return loss.detach().cpu().numpy()

    def prep_input(self, x):
        x = x.float().to(self.device)
        # (b, t, m, n, d) -> (b, t, d, m, n)
        x = x.permute(0, 1, 4, 2, 3)
        return x

