import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torchmetrics.functional import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


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
        self.metric_names = ["MSE", "MAE", "MAPE", "RMSE"]

    def train(self, model, batch_generator):
        model = model.to(self.device)
        model.train()

        optimizer = self.__get_optimizer(model)
        train_loss, val_loss = [], []
        tolerance, best_epoch, best_val_loss, best_train_loss = 0, 0, 1e6, 1e6
        best_train_metric, best_val_metric = None, None
        best_dict = model.state_dict()
        for epoch in range(self.num_epochs):
            # train and validation loop
            start_time = time.time()
            running_train_loss, train_metric_scores = self.__step_loop(model=model,
                                                                       generator=batch_generator,
                                                                       mode='train',
                                                                       optimizer=optimizer)
            running_val_loss, val_metric_scores = self.__step_loop(model=model,
                                                                   generator=batch_generator,
                                                                   mode='val',
                                                                   optimizer=None)
            epoch_time = time.time() - start_time

            # save the losses
            train_loss.append(running_train_loss)
            val_loss.append(running_val_loss)

            # log the scores
            train_metric_str = self.get_metric_string(metric_scores=train_metric_scores)
            val_metric_str = self.get_metric_string(metric_scores=val_metric_scores)
            print(f"\t --> Epoch:{epoch + 1}/{self.num_epochs} took {epoch_time:.3f} secs:\t"
                  f"Train_loss: {running_train_loss:.5f}, {train_metric_str}\t "
                  f"Val_loss: {running_val_loss:.5f}, {val_metric_str}")

            # early stopping criteria
            if running_val_loss < best_val_loss:
                best_dict = deepcopy(model.state_dict())
                best_epoch = epoch + 1
                best_val_loss = running_val_loss
                best_val_metric = val_metric_scores
                best_train_loss = running_train_loss
                best_train_metric = train_metric_scores
                tolerance = 0
            else:
                tolerance += 1

            if tolerance > self.tolerance or epoch == self.num_epochs - 1:
                model.load_state_dict(best_dict)
                train_metric_str = self.get_metric_string(metric_scores=best_train_metric)
                val_metric_str = self.get_metric_string(metric_scores=best_val_metric)
                print(f"\tEarly exiting from epoch: {best_epoch}:\t"
                      f"Train_loss {best_train_loss:.5f}, {train_metric_str}\t"
                      f"Validation_loss: {best_val_loss:.5f}, {val_metric_str}")
                break
            torch.cuda.empty_cache()

        return (train_loss, val_loss), best_train_metric, best_val_metric

    def evaluate(self, model, batch_generator):
        model = model.to(self.device)

        # train the model on the set of train+eval with the
        # number of epochs that is divided by 4
        num_epochs = max(self.num_epochs // 4, 1)
        optimizer = self.__get_optimizer(model)
        for epoch in range(num_epochs):
            # train + val
            start_time = time.time()
            running_eval_loss, running_metric_scores = self.__step_loop(model=model,
                                                                        generator=batch_generator,
                                                                        mode='train_val',
                                                                        optimizer=optimizer)
            epoch_time = time.time() - start_time

            metric_str = self.get_metric_string(metric_scores=running_metric_scores)
            print(f'\t --> Epoch:{epoch + 1}/{num_epochs} took {epoch_time:.3f} secs:\t'
                  f'Eval_loss:{running_eval_loss:.5f}, {metric_str}')

        model.eval()
        with torch.no_grad():
            eval_loss, eval_metric_scores = self.__step_loop(model=model,
                                                             generator=batch_generator,
                                                             mode='train_val',
                                                             optimizer=None)
            metric_str = self.get_metric_string(metric_scores=eval_metric_scores)
            print(f"\tEvaluation finished: Loss: {eval_loss:.5f}, {metric_str}")
        torch.cuda.empty_cache()

        return eval_loss, eval_metric_scores

    def predict(self, model, batch_generator):
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            test_loss, test_metric_scores = self.__step_loop(model=model,
                                                             generator=batch_generator,
                                                             mode='test',
                                                             optimizer=None)

            metric_str = self.get_metric_string(metric_scores=test_metric_scores)
            print(f"\tTest finished: Loss: {test_loss:.5f}, {metric_str}")
        torch.cuda.empty_cache()

        return test_loss, test_metric_scores

    def __step_loop(self, model, generator, mode, optimizer):
        if mode in ['test', 'val']:
            step_fun = self.__val_step
        else:
            step_fun = self.__train_step

        running_loss, running_metric_scores = 0, {key: 0 for key in self.metric_names}
        for idx, (x, y) in enumerate(generator.generate(mode)):
            print('\r\t{}:{}/{}'.format(mode, idx, generator.num_iter(mode)),
                  flush=True, end='')

            if hasattr(model, 'hidden'):
                hidden = model.init_hidden(batch_size=x.shape[0])
            else:
                hidden = None

            x, y = [self.__prep_input(i) for i in [x, y]]
            loss, metric_scores = step_fun(model=model,
                                           inputs=[x, y, hidden],
                                           optimizer=optimizer,
                                           generator=generator)
            running_loss += loss
            for key, score in metric_scores.items():
                running_metric_scores[key] += score

        running_loss /= (idx + 1)
        for key, score in running_metric_scores.items():
            running_metric_scores[key] = score / (idx + 1)

        return running_loss, running_metric_scores

    def __train_step(self, model, inputs, optimizer, generator):
        x, y, hidden = inputs
        if optimizer is not None:
            optimizer.zero_grad()
        pred = model.forward(x=x, hidden=hidden)
        loss = self.criterion(pred, y)

        if model.is_trainable and optimizer is not None:
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), self.clip)

            # take step in classifier's optimizer
            optimizer.step()

        loss, metric_scores = self.__calc_scores(pred, y, generator)

        return loss, metric_scores

    def __val_step(self, model, inputs, optimizer, generator):
        x, y, hidden = inputs
        pred = model.forward(x=x, hidden=hidden)
        loss, metric_scores = self.__calc_scores(pred, y, generator)

        return loss, metric_scores

    def __prep_input(self, x):
        x = x.float().to(self.device)
        # (b, t, m, n, d) -> (b, t, d, m, n)
        x = x.permute(0, 1, 4, 2, 3)
        return x

    def __get_optimizer(self, model):
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

        return optimizer

    def __calc_scores(self, pred, y, generator):
        if generator.normalizer:
            pred = generator.normalizer.inv_norm(pred, self.device)
            y = generator.normalizer.inv_norm(y, self.device)

        metric_scores = self.__calc_metrics(pred=pred, y=y)
        loss = self.criterion(pred, y).detach().cpu().numpy()

        return loss, metric_scores

    @staticmethod
    def __calc_metrics(pred, y):
        metric_collection = {
            "MSE": mean_squared_error,
            "MAE": mean_absolute_error,
            "MAPE": mean_absolute_percentage_error,
            "RMSE": lambda preds, target: torch.sqrt(mean_squared_error(preds, target))
        }

        metric_scores = {}
        for key, metric_fun in metric_collection.items():
            metric_scores[key] = metric_fun(preds=pred, target=y).detach().cpu().numpy()

        return metric_scores

    @staticmethod
    def get_metric_string(metric_scores):
        message = ""
        for key, score in metric_scores.items():
            message += f"{key}: {score:.5f}, "

        return message
