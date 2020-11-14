experiment_params = {
    "global_start_date": '2000-01-01',
    "global_end_date": '2008-12-31',
    "data_step": 3,
    "data_length": 12,
    "val_ratio": 0.1
}

data_params = {
    "weather_raw_dir": 'data/data_raw',
    "spatial_range": [],  # [[30, 45], [20, 50]],
    "weather_freq": 3,
    "check_files": False,
    "features": ['d', 'cc', 'z', 'pv', 'r', 'ciwc', 'clwc', 'q', 'crwc', 'cswc', 't', 'u', 'v', 'w'],
    "atm_dim": -1,
    "rebuild": False
}

batch_gen_params = {
    "input_dim": list(range(14)),
    "output_dim": [10],
    "window_in_len": 10,
    "window_out_len": 5,
    "batch_size": 16
}


trainer_params = {
    "global_start_date": '2000-01-01',
    "global_end_date": '2008-12-31',
}
