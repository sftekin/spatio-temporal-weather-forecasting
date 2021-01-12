experiment_params = {
    "global_start_date": '2000-01-01',
    "global_end_date": '2008-12-31',
    "data_step": 6,
    "data_length": 24,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "normalize_flag": True,
    "model": "weather_model",
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
    "window_in_len": 10,
    "window_out_len": 10,
    "batch_size": 8,
    "shuffle": True,
}

trainer_params = {
    "num_epochs": 100,
    "learning_rate": 0.0007,
    "clip": 5,
    "early_stop_tolerance": 4
}

model_params = {
    "moving_avg": {
        "window_in": 30,
        "window_out": 10,
        "output_dim": 5,
        "mode": "WMA"
    },
    "convlstm": {
        "input_size": (61, 121),
        "window_in": 10,
        "window_out": 10,
        "num_layers": 3,
        "encoder_params": {
            "input_dim": 9,
            "hidden_dims": [1, 16, 32],
            "kernel_size": [3, 3, 3],
            "bias": False,
            "peephole_con": False
        },
        "decoder_params": {
            "input_dim": 32,
            "hidden_dims": [32, 16, 1],
            "kernel_size": [3, 3, 3],
            "bias": False,
            "peephole_con": False
        }
    },
    "u_net": {
        "selected_dim": 5,
        "in_channels": 10,
        "out_channels": 10
    },
    "weather_model": {
        "input_size": (61, 121),
        "window_in": 10,
        "window_out": 10,
        "num_layers": 3,
        "selected_dim": 5,
        "attention_params": {
            "input_size": (61, 121),
            "input_dim": 10,
            "hidden_dim": 1,
            "attn_dim": 300
        },
        "encoder_params": {
            "input_dim": 9,
            "hidden_dims": [1, 16, 32],
            "kernel_size": [3, 3, 3],
            "bias": False,
            "peephole_con": False
        },
        "decoder_params": {
            "input_dim": 32,
            "hidden_dims": [32, 16, 1],
            "kernel_size": [3, 3, 3],
            "bias": False,
            "peephole_con": False
        }
    }
}
