import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class Trainer:
    def __init__(self, num_epochs, early_stop_tolerance, clip, optimizer,
                 learning_rate, weight_decay, momentum, device):
        self.num_epochs = num_epochs
        self.clip = clip
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.tolerance = early_stop_tolerance
        self.device = torch.device(device)
        self.criterion = nn.MSELoss()

    def fit(self, model, batch_generator):
        model = model.to(self.device)
        model.train()

        if model.is_trainable:
            if self.optimizer == "adam":
                optimizer = optim.Adam(model.parameters(),
                                       lr=self.learning_rate)
            else:
                optimizer = optim.SGD(model.parameters(),
                                      lr=self.learning_rate,
                                      momentum=self.momentum)
        else:
            optimizer = None

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
            running_train_loss = self.__step_loop(model=model,
                                                  generator=batch_generator,
                                                  mode='train',
                                                  optimizer=optimizer)

            # validation
            running_val_loss = self.__step_loop(model=model,
                                                generator=batch_generator,
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

                evaluation_val_loss = self.__step_loop(model=model,
                                                       generator=batch_generator,
                                                       mode='val',
                                                       optimizer=None)

                message_str = "Early exiting from epoch: {}, Validation error: {:.5f}."
                print(message_str.format(best_epoch, evaluation_val_loss))
                break

            torch.cuda.empty_cache()

        print('Train finished, best eval lost: {:.5f}'.format(evaluation_val_loss))
        return train_loss, val_loss, evaluation_val_loss

    def transform(self, model, batch_generator):
        test_loss = self.__step_loop(model=model,
                                     generator=batch_generator,
                                     mode='test',
                                     optimizer=None)
        print('Test finished, best eval lost: {:.5f}'.format(test_loss))
        return test_loss

    def __step_loop(self, model, generator, mode, optimizer):
        running_loss = 0
        batch_size = generator.dataset_params['batch_size']
        if mode in ['test', 'val']:
            step_fun = self.__val_step
        else:
            step_fun = self.__train_step
        idx = 0
        for idx, (x, y, f_x) in enumerate(generator.generate(mode)):
            print('\r{}:{}/{}'.format(mode, idx, generator.num_iter(mode)),
                  flush=True, end='')

            if hasattr(model, 'hidden'):
                hidden = model.init_hidden(batch_size)
            else:
                hidden = None

            x, y = [self.__prep_input(i) for i in [x, y]]
            loss = step_fun(model=model,
                            inputs=[x, y, f_x.float().to(self.device), hidden],
                            optimizer=optimizer,
                            generator=generator)

            running_loss += loss
        running_loss /= (idx + 1)

        return running_loss

    def __train_step(self, model, inputs, optimizer, generator):
        x, y, f_x, hidden = inputs
        if optimizer:
            optimizer.zero_grad()
        pred = model.forward(x=x, f_x=f_x, hidden=hidden)
        loss = self.criterion(pred, y)

        if model.is_trainable:
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), self.clip)

            # take step in classifier's optimizer
            optimizer.step()

        if generator.normalizer:
            pred = generator.normalizer.inv_norm(pred, self.device)
            y = generator.normalizer.inv_norm(y, self.device)

        de_norm_loss = self.criterion(pred, y)
        de_norm_loss = de_norm_loss.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        print(f"  loss: {de_norm_loss}")

        return de_norm_loss

    def __val_step(self, model, inputs, optimizer, generator):
        x, y, f_x, hidden = inputs
        pred = model.forward(x=x, f_x=f_x, hidden=hidden)
        if generator.normalizer:
            pred = generator.normalizer.inv_norm(pred, self.device)
            y = generator.normalizer.inv_norm(y, self.device)

        loss = self.criterion(pred, y)

        return loss.detach().cpu().numpy()

    def __prep_input(self, x):
        x = x.float().to(self.device)
        # (b, t, m, n, d) -> (b, t, d, m, n)
        x = x.permute(0, 1, 4, 2, 3)
        return x
