import os
import glob
import pickle as pkl

from trainer import Trainer


def train(model_name, model, batch_generator, trainer_params, device):
    experiment_count = _get_exp_count()
    save_dir = os.path.join('results', model_name, 'exp_' + str(experiment_count+1))
    os.makedirs(save_dir)

    loss_save_path = os.path.join(save_dir, 'loss.pkl')
    model_save_path = os.path.join(save_dir, 'model.pkl')
    trainer_save_path = os.path.join(save_dir, 'trainer.pkl')

    trainer = Trainer(device=device, **trainer_params)
    train_val_loss = trainer.fit(model, batch_generator)
    test_loss = trainer.transform(model, batch_generator)

    train_val_loss += (test_loss,)

    for path, obj in zip([loss_save_path, model_save_path, trainer_save_path],
                         [train_val_loss, model, trainer]):
        with open(path, 'wb') as file:
            pkl.dump(obj, file)


def predict(model_name, batch_generator):
    # find the best model
    model, trainer = _find_best_model(model_name)

#    trainer.fit_batch_generator(batch_generator)
    predict_loss = trainer.transform(model, batch_generator)

    return predict_loss


def _find_best_model(model_name):
    print("Selecting best model...")
    exps_dir = os.path.join('results', model_name)

    best_model = None
    best_loss = 1e6
    for exp in glob.glob(os.path.join(exps_dir, 'exp_*')):
        loss_path = os.path.join(exp, 'loss.pkl')
        model_path = os.path.join(exp, 'model.pkl')
        trainer_path = os.path.join(exp, 'trainer.pkl')
        
        try:
            with open(loss_path, 'rb') as f:
                loss = pkl.load(f)
            best_eval_loss = loss[-1]
            if best_eval_loss < best_loss:
                best_loss = best_eval_loss
                with open(model_path, 'rb') as f:
                    best_model = pkl.load(f)
                with open(trainer_path, 'rb') as f:
                    trainer = pkl.load(f)
        except Exception as e:
            pass
    return best_model, trainer


def _load_model(experiment_num):
    exp = f"results/exp_{experiment_num}"
    loss_path = os.path.join(exp, 'loss.pkl')
    model_path = os.path.join(exp, 'model.pkl')
    trainer_path = os.path.join(exp, 'trainer.pkl')

    with open(loss_path, 'rb') as f:
        loss = pkl.load(f)

    with open(model_path, 'rb') as f:
        model = pkl.load(f)

    with open(trainer_path, 'rb') as f:
        trainer = pkl.load(f)

    return model, trainer


def _get_exp_count():
    save_dir = 'results'
    num_exp_dir = len(glob.glob(os.path.join(save_dir, 'exp_*')))
    return num_exp_dir
