data_params = {
    "weather_raw_dir": '/media/selim/data/dataset/ecmwf/atmosphere',
    "start_date": '1994-01-01',
    "end_date": '1999-12-31',
    "spatial_range": [[37.585, 43.835], [-77.125, -70.875]],
    "weather_freq": 3,
    "check_files": False,
    "rebuild": False
}

dataset_params = {
    "input_dim": list(range(5)),
    "output_dim": [0],
    "atm_dim": 0,
    "window_len": 10,
    "batch_size": 16
}

batch_gen_params = {
    "test_ratio": 0.1,
    "val_ratio": 0.1
}
