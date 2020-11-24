from dataset import WeatherDataset
from models.adaptive_normalizer import AdaptiveNormalizer


class BatchGenerator:
    def __init__(self, weather_data, val_ratio, test_ratio, normalize_flag, params):
        self.weather_data = weather_data
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.dataset_params = params
        self.normalize_flag = normalize_flag

        if self.normalize_flag:
            self.normalizer = AdaptiveNormalizer(output_dim=params['output_dim'])
        else:
            self.normalizer = None

        self.weather_dict = self.__split_data(self.weather_data)
        self.dataset_dict = self.__create_sets()

    def __split_data(self, in_data):
        data_len = len(in_data)
        val_count = int(data_len * self.val_ratio)
        test_count = int(data_len * self.val_ratio)

        train_count = data_len - val_count - test_count

        data_dict = {
            'train': in_data[:train_count],
            'val': in_data[train_count:train_count+val_count],
            'test': in_data[train_count+val_count:]
        }

        return data_dict

    def __create_sets(self):
        hurricane_dataset = {}
        for i in ['train', 'val', 'test']:
            dataset = WeatherDataset(weather_data=self.weather_dict[i],
                                     normalizer=self.normalizer,
                                     **self.dataset_params)
            hurricane_dataset[i] = dataset

        return hurricane_dataset

    def num_iter(self, dataset_name):
        return self.dataset_dict[dataset_name].num_iter

    def generate(self, dataset_name):
        selected_loader = self.dataset_dict[dataset_name]
        yield from selected_loader.next()
