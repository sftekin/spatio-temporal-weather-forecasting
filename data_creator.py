import os
import pandas as pd
import pickle as pkl

from transformer.weather_transform import WeatherTransformer


class DataCreator:
    def __init__(self, weather_raw_dir, start_date, end_date, spatial_range,
                 weather_freq=3, check_files=False, rebuild=True):
        """
        Creates weather data. Stores path of the each data file as list
        under `self.weather_data` attribute.

        :param str weather_raw_dir:
        :param str start_date: e.g '2018-01-10'
        :param str end_date: e.g '2019-01-12'
        :param list of list spatial_range: e.g [[40, 43], [-96, -89]]
        :param int weather_freq:
        :param bool check_files:
        :param bool rebuild:
        """
        # Data paths
        self.data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.data_dir, 'data')
        self.weather_raw_dir = weather_raw_dir

        # Parameters
        self.start_date = start_date
        self.end_date = end_date
        self.spatial_range = spatial_range
        self.weather_freq = weather_freq
        self.check_files = check_files
        self.rebuild = rebuild

        # create the data
        self.weather_data = self.__create_data()

    def __create_data(self):
        weather_folder = os.path.join(self.data_dir, 'weather_data')

        if not self.rebuild:
            print('Loading from saved path')
            weather_list = []
            for weather_file in os.listdir(weather_folder):
                weather_file_path = os.path.join(weather_folder, weather_file)
                weather_list.append(weather_file_path)

            if len(weather_list) == 0:
                raise ValueError('{} folder is empty'.format(weather_folder))

        else:
            # create the weather_data folder
            if not os.path.isdir(weather_folder):
                os.makedirs(weather_folder)

            print('Creating the weather data')
            weather_transformer = WeatherTransformer(file_dir=self.weather_raw_dir,
                                                     check=self.check_files)

            # create weather data
            date_r = pd.date_range(start=self.start_date,
                                   end=self.end_date,
                                   freq=str(self.weather_freq) + 'H')

            weather_data = weather_transformer.transform_range(date_range=date_r,
                                                               spatial_range=self.spatial_range)

            weather_data_path = os.path.join(weather_folder, 'weather_data.pkl')
            with open(weather_data_path, 'wb') as f:
                pkl.dump(weather_data, f)


if __name__ == '__main__':
    # /media/selim/data/dataset/ecmwf/atmosphere
    pass
