data_params = {
    "weather_raw_dir": '/media/NAS/DataScienceShared/meteorology/reanalysis/pressure',
    "start_date": '2000-01-01',
    "end_date": '2002-12-31',
    "spatial_range": [[30, 45], [20, 50]],
    "weather_freq": 1,
    "check_files": False,
    "features": ['d', 'cc', 'z', 'pv', 'r', 'ciwc', 'clwc', 'q', 'crwc', 'cswc', 't', 'u', 'v', 'w'],
    "atm_dim": -1,
    "rebuild": True
}

dataset_params = {
    "input_dim": list(range(5)),
    "output_dim": [0],
    "window_len": 10,
    "batch_size": 16
}

batch_gen_params = {
    "test_ratio": 0.1,
    "val_ratio": 0.1
}

trainer_params = {

}
