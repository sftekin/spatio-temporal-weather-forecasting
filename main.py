from config import data_params, batch_params
from data_creator import DataCreator
from dataset import WeatherDataset


def main():
    data_creator = DataCreator(**data_params)
    dataset = WeatherDataset(weather_data=data_creator.weather_data, **batch_params)

    for x, y in dataset.next():
        print(x.shape, y.shape)


if __name__ == '__main__':
    main()

