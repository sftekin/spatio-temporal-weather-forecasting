import os
import shutil
import pandas as pd

from config import experiment_params, data_params, batch_gen_params
from data_creator import DataCreator
from batch_generator import BatchGenerator


def run():
    global_start_date = experiment_params['global_start_date']
    global_end_date = experiment_params['global_end_date']
    data_step = experiment_params['data_step']
    data_length = experiment_params['data_length']
    val_ratio = experiment_params['val_ratio']

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
                                         params=batch_gen_params)

        for c, (x, y) in enumerate(batch_generator.generate(dataset_name='train')):
            print(c, x.shape, y.shape)

        # # remove dump directory
        # shutil.rmtree(dump_file_dir)


if __name__ == '__main__':
    run()

