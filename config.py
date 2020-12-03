experiment_params = {
    "global_start_date": '2000-01-01',
    "global_end_date": '2008-12-31',
    "data_step": 3,
    "data_length": 24,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "normalize_flag": True,
    "model": "moving_avg",
    "device": 'cuda'
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
    "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],
    "output_dim": 10,
    "window_in_len": 30,
    "window_out_len": 10,
    "batch_size": 8,
}

trainer_params = {
    "num_epochs": 30,
    "learning_rate": 0.001,
    "clip": 5,
    "early_stop_tolerance": 4
}

model_params = {
    "moving_avg": {
        "window_in": 30,
        "window_out": 10,
        "output_dim": 5,
        "mode": "EMA"
    },
    "weather_model": {
        "input_size": (61, 121),
        "window_in": 30,
        "window_out": 10,
        "num_series": 9,
        "output_dim": 5,
        "input_attn_dim": 200,
        "encoder_params": {
            "hidden_dim": 16,
            "flow_dim": 4,
            "kernel_size": 3,
            "bias": False,
            "padding": 2,
        },
        "decoder_params": {
            "hidden_dim": 16,
            "flow_dim": 4,
            "kernel_size": 3,
            "padding": 2,
            "bias": False
        }
    }
}
