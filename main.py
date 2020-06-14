from config import data_params, dataset_params, batch_gen_params
from data_creator import DataCreator
from batch_generator import BatchGenerator


def main():
    data_creator = DataCreator(**data_params)
    batch_generator = BatchGenerator(weather_data=data_creator.weather_data,
                                     val_ratio=batch_gen_params['val_ratio'],
                                     test_ratio=batch_gen_params['test_ratio'],
                                     dataset_params=dataset_params)

    for x, y in batch_generator.generate(dataset='train'):
        print(x.shape, y.shape)


if __name__ == '__main__':
    main()

