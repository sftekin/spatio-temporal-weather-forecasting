from dataset import WeatherDataset


class BatchGenerator:
    def __init__(self, weather_data, val_ratio, params):
        self.weather_data = weather_data
        self.val_ratio = val_ratio
        self.dataset_params = params

        self.weather_dict = self.__split_data(self.weather_data)
        self.dataset_dict = self.__create_sets()

    def __split_data(self, in_data):
        data_len = len(in_data)
        val_count = int(data_len * self.val_ratio)
        train_count = data_len - val_count

        data_dict = {
            'train': in_data[:train_count],
            'val': in_data[train_count+val_count:]
        }

        return data_dict

    def __create_sets(self):
        hurricane_dataset = {}
        for i in ['train', 'val']:
            dataset = WeatherDataset(weather_data=self.weather_dict[i], **self.dataset_params)
            hurricane_dataset[i] = dataset

        return hurricane_dataset

    def generate(self, dataset_name):
        selected_loader = self.dataset_dict[dataset_name]
        yield from selected_loader.next()


