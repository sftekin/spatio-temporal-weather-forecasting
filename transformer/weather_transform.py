import os
import gc
import netCDF4
import numpy as np
import pandas as pd


class WeatherTransformer:
    def __init__(self, file_dir, check=False):
        self.file_dir = file_dir
        self.features = ['d', 'cc', 'z', 'u', 'v']
        self.index_date = pd.to_datetime('1900-01-01')
        self.freq = 3

        self.dates = self._get_file_dates()
        if check:
            self._check_filename_date_matching()

    def _get_file_dates(self):
        file_names = os.listdir(self.file_dir)
        file_names = list(map(lambda x: x.split('.')[0].replace('_', '-') + '-01', file_names))
        file_dates = pd.to_datetime(file_names)

        return file_dates

    def _check_filename_date_matching(self):
        for file_name in os.listdir(self.file_dir):
            file_path = os.path.join(self.file_dir, file_name)
            nc = netCDF4.Dataset(file_path, 'r')
            first_date = int(nc['time'][:][0])

            first_date = self.index_date + pd.DateOffset(hours=first_date)

            file_ = file_name.split('.')[0]
            file_y, file_m = file_.split('_')

            if first_date.year != int(file_y) or first_date.month != int(file_m):
                raise IndexError('{} does not match with inside date'.format(file_))
            gc.collect()

    def transform_range(self, date_range, spatial_range, save_file=None):
        """

        :param date_range: e.g pd.date_range(start='2019-01-01', end='2020-03-01', freq='3H')
        :param spatial_range: e.g [[40, 43], [-96, -89]
        :return:
        """
        if date_range.freq.n != self.freq:
            raise ValueError('Input date_range must have the '
                             'same frequency with netCDF files')

        file_dates = []
        for day in date_range:
            year = str(day.year)
            if day.month < 10:
                month = '0' + str(day.month)
            else:
                month = str(day.month)
            file_name = year + '_' + month + '.nc'
            file_dates.append(file_name)
        file_dates = np.array(file_dates)
        _, idx = np.unique(file_dates, return_index=True)
        file_dates = file_dates[np.sort(idx)]

        print('Spatial cropping started')
        time_arr_list = []
        data_arr_list = []
        for count, file_name in enumerate(file_dates):
            print(r'{:.2f}%'.format((count / len(file_dates)) * 100), flush=True, end='\r')
            file_path = os.path.join(self.file_dir, file_name)
            nc = netCDF4.Dataset(file_path, 'r')

            time_arr = np.array(nc['time'][:], dtype=np.int)
            time_arr_list.append(time_arr)

            data_arr = self._crop_spatial(data=nc, in_range=spatial_range)

            # TODO: save data for each timestamp
            if save_file:
                pass

            data_arr_list.append(data_arr)
            # since files are big, garbage collect the unref. files
            gc.collect()

        # combine all arrays on time dimension
        data_combined = np.concatenate(data_arr_list, axis=0)
        time_combined = np.concatenate(time_arr_list, axis=0)

        # temporal crop
        temporal_idx = self._crop_temporal(time_combined, date_range)
        data_cropped = data_combined[temporal_idx]

        return data_cropped

    def _crop_spatial(self, data, in_range):
        """
        :param data:
        :param in_range:[[40, 43], [-96, -89]
        :return:
        """
        lats = data['latitude'][:]
        lons = data['longitude'][:]
        lat_bnds, lon_bnds = in_range

        lat_inds = np.where((lats > lat_bnds[0]) & (lats < lat_bnds[1]))[0]
        lon_inds = np.where((lons > lon_bnds[0]) & (lons < lon_bnds[1]))[0]

        arr_list = []
        for key in self.features:
            subset_arr = data.variables[key][:, :, lat_inds, lon_inds]
            arr_list.append(np.array(subset_arr))

        # combine all features to create arr with shape of
        # (T, L, M, N, D) where L is the levels
        data_combined = np.stack(arr_list, axis=4)

        return data_combined

    def _crop_temporal(self, time_arr, date_range):
        fun = np.vectorize(lambda x: self.index_date + pd.DateOffset(hours=int(x)))
        in_date_range = pd.to_datetime(fun(time_arr))

        start_date, end_date = date_range[0], date_range[-1]
        indices = (start_date <= in_date_range) & (in_date_range <= end_date)

        return indices


if __name__ == '__main__':
    weather_tf = WeatherTransformer(check=False)

    date_r = pd.date_range(start='2018-01-10', end='2019-01-12', freq='3H')
    print('Total time length: ', len(date_r))
    spat_r = [[40, 43], [-96, -89]]

    ret_arr = weather_tf.transform(date_range=date_r, spatial_range=spat_r)
    print(ret_arr.shape)
