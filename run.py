import os
import shutil
import pandas as pd
import pickle as pkl

from config import experiment_params, data_params, batch_gen_params, trainer_params, model_params
from data_creator import DataCreator
from batch_generator import BatchGenerator
from trainer import Trainer
from models.weather_model import WeatherModel


def run():
    global_start_date = experiment_params['global_start_date']
    global_end_date = experiment_params['global_end_date']
    data_step = experiment_params['data_step']
    data_length = experiment_params['data_length']
    val_ratio = experiment_params['val_ratio']
    normalize_flag = experiment_params['normalize_flag']
    device = experiment_params['device']

    save_dir = 'results'

    dump_file_dir = os.path.join(data_params['weather_raw_dir'], 'data_dump')

    months = pd.date_range(start=global_start_date, end=global_end_date, freq=str(1) + 'M')
    for i in range(0, len(months) - (data_length - data_step), data_step):
        start_date_str = '-'.join([str(months[i].year), str(months[i].month), '01'])
        start_date = pd.to_datetime(start_date_str)
        end_date = start_date + pd.DateOffset(months=data_length) - pd.DateOffset(hours=1)

        data_creator = DataCreator(start_date=start_date, end_date=end_date, **data_params)
        weather_data = data_creator.create_data()

        batch_generator = BatchGenerator(weather_data=weather_data,
                                         val_ratio=val_ratio,
                                         params=batch_gen_params,
                                         normalize_flag=normalize_flag)

        model = WeatherModel(device=device, **model_params)
        trainer = Trainer(batch_generator=batch_generator, device=device, **trainer_params)

        loss = trainer.fit(model)

        loss_save_path = os.path.join(save_dir, 'loss.pkl')
        model_save_path = os.path.join(save_dir, 'model.pkl')
        trainer_save_path = os.path.join(save_dir, 'trainer.pkl')
        for path, obj in zip([loss_save_path, model_save_path, trainer_save_path],
                             [loss, model, trainer]):
            with open(path, 'wb') as file:
                pkl.dump(obj, file)

        break

        # # remove dump directory
        # shutil.rmtree(dump_file_dir)


if __name__ == '__main__':
    run()

