import os
import shutil
import pandas as pd

from config import experiment_params, data_params, batch_gen_params, trainer_params, model_params
from data_creator import DataCreator
from batch_generator import BatchGenerator
from experiment import train, predict
from models.weather_model import WeatherModel


def run():
    global_start_date = experiment_params['global_start_date']
    global_end_date = experiment_params['global_end_date']
    stride = experiment_params['data_step']
    data_length = experiment_params['data_length']
    val_ratio = experiment_params['val_ratio']
    test_ratio = experiment_params['test_ratio']
    normalize_flag = experiment_params['normalize_flag']
    device = experiment_params['device']

    dump_file_dir = os.path.join(data_params['weather_raw_dir'], 'data_dump')

    months = pd.date_range(start=global_start_date, end=global_end_date, freq=str(1) + 'M')
    for i in range(0, len(months) - (data_length - stride), stride):
        start_date_str = '-'.join([str(months[i].year), str(months[i].month), '01'])
        start_date = pd.to_datetime(start_date_str)
        end_date = start_date + pd.DateOffset(months=data_length) - pd.DateOffset(hours=1)

        data_creator = DataCreator(start_date=start_date, end_date=end_date, **data_params)
        weather_data = data_creator.create_data()

        batch_generator = BatchGenerator(weather_data=weather_data,
                                         val_ratio=val_ratio,
                                         test_ratio=test_ratio,
                                         params=batch_gen_params,
                                         normalize_flag=normalize_flag)

        model = WeatherModel(device=device, **model_params)

        train(model=model, batch_generator=batch_generator, trainer_params=trainer_params, device=device)
        predict(batch_generator=batch_generator)

        break

        # # remove dump directory
        # shutil.rmtree(dump_file_dir)


if __name__ == '__main__':
    run()

