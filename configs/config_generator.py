import copy
import itertools


class ConfigGenerator:
    def conf_next(self, input_conf):
        for conf in self.__grid_search(input_conf):
            yield conf

    def __grid_search(self, in_conf):
        copy_conf = copy.deepcopy(in_conf)
        keys, combinations = self.__get_combinations(in_conf=copy_conf)
        if combinations:
            for param in combinations:
                self.__update_config(in_dict=copy_conf, keys=keys, values=param)
                yield copy_conf
        else:
            yield copy_conf

    def __get_combinations(self, in_conf):
        params = self.__search_params(in_dict=in_conf)
        combinations = []
        keys = ()
        if params:
            keys, values = zip(*params)
            for param in itertools.product(*values):
                combinations.append(param)
        return keys, combinations

    def __search_params(self, in_dict):
        dynamic_param_list = []
        dict_elements = [
            val for val in in_dict.values() if isinstance(val, dict)
        ]

        if dict_elements:
            for element in dict_elements:
                dynamic_param_list += self.__search_params(in_dict=element)

        for key, val in in_dict.items():
            if isinstance(val, Param):
                dynamic_param_list.append((key, val.data))

        return dynamic_param_list

    def __update_config(self, in_dict, keys, values):
        """
        update input dictionary, 'in_dict' by given keys and values

        :param dict in_dict: Config dictionary
        :param tuple keys: (key, key)
        :param tuple values:(value, value)
        :return: None
        :rtype: None
        """
        for key, val in zip(keys, values):
            self.__change_value_in_dict(in_dict=in_dict, in_key=key, in_value=val)

    def __change_value_in_dict(self, in_dict, in_key, in_value):
        """
        find 'in_key' in 'in_dict' changes value of 'in_key' to 'in_value'
        works in nested dicts.

        :param dict in_dict: Config dictionary
        :param str in_key: Initial key
        :param Any in_value: any data type
        :return: Initial dictionary
        :rtype: dict
        """
        for key, value in in_dict.items():
            if key == in_key:
                in_dict[key] = in_value
            elif isinstance(value, dict):
                value = self.__change_value_in_dict(in_dict=value, in_key=in_key, in_value=in_value)
                in_dict[key] = value

        return in_dict


class Param:
    def __init__(self, in_data):
        if not isinstance(in_data, list):
            self.data = [in_data]
        else:
            self.data = in_data


if __name__ == '__main__':
    sample_conf = {
        "input_size": (61, 121),
        "window_in": 10,
        "window_out": 5,
        "num_layers": 1,
        "encoder_params": {
            "input_dim": 9,
            "hidden_dims": Param([1, 3, 4]),
            "kernel_size": [3],
            "bias": False,
            "peephole_con": False
        },
        "decoder_params": {
            "input_dim": 1,
            "hidden_dims": [1],
            "kernel_size": [3],
            "bias": False,
            "peephole_con": False
        }
    }

    conf_generator = ConfigGenerator()
    for conf in conf_generator.conf_next(sample_conf):
        print(conf)
