import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model_name, batch_generator, trainer_params):
    experiment_count = _get_exp_count()

    save_dir = 'experiment_' + str(experiment_count)
    os.path.join('results', save_dir)

    # create experiment directory if it isn't created before
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)







def _get_exp_count():
    exp_list = []
    for path in os.listdir('results'):
        if 'exp' in path:
            exp_str = path.split('_')[-1]
            exp_list.append(int(exp_str))

    if len(exp_list) == 0:
        exp_count = 0
    else:
        exp_count = max(exp_list)

    return exp_count



