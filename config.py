from config_generator import Param

experiment_params = {
    "global_start_date": '2000-01-01',
    "global_end_date": '2008-12-31',
    "data_step": 6,
    "data_length": 12,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "normalize_flag": True,
    "model": "traj_gru",
    "device": 'cuda',
    "selected_criterion": "MSE"  # choices are MSE, MAE, and MAPE
}

data_params = {
    "weather_raw_dir": 'data/data_raw',
    "spatial_range": [],  # [[30, 45], [20, 50]],
    "weather_freq": 3,
    "downsample_mode": "selective",  # can be average or selective
    "check_files": False,
    "features": ['d', 'cc', 'z', 'pv', 'r', 'ciwc', 'clwc', 'q', 'crwc', 'cswc', 't', 'u', 'v', 'w'],
    "atm_dim": -1,
    "target_dim": 10,
    "smooth": False,
    "smooth_win_len": 31,  # select odd
    "rebuild": False
}

model_params = {
    "moving_avg": {
        "batch_gen": {
            "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],
            "output_dim": 10,
            "window_in_len": 10,
            "window_out_len": 5,
            "batch_size": 8,
            "shuffle": True,
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 4
        },
        "core": {
            "window_in": 10,  # should be same with batch_gen["window_in_len"]
            "window_out": 5,  # should be same with batch_gen["window_out_len"]
            "output_dim": 5,
            "mode": "WMA"
        }
    },
    "convlstm": {
        "batch_gen": {
            "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],
            "output_dim": 10,
            "window_in_len": 10,
            "window_out_len": 5,
            "batch_size": 8,
            "shuffle": True,
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 4
        },
        "core": {
            "input_size": (61, 121),
            "window_in": 10,  # should be same with batch_gen["window_in_len"]
            "window_out": 5,  # should be same with batch_gen["window_out_len"]
            "num_layers": 1,
            "encoder_params": {
                "input_dim": 9,
                "hidden_dims": [1],
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
        },
    },
    "u_net": {
        "batch_gen": {
            "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],
            "output_dim": 10,
            "window_in_len": 10,
            "window_out_len": 5,
            "batch_size": 8,
            "shuffle": True,
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 6
        },
        "core": {
            "selected_dim": 5,
            "in_channels": 10,  # should be same with batch_gen["window_in_len"]
            "out_channels": 5  # should be same with batch_gen["window_out_len"]
        }

    },
    "weather_model": {
        "batch_gen": {
            "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],  # indexes of selected features for input
            "output_dim": 10,  # indexes of selected features for output
            "window_in_len": 10,
            "window_out_len": 5,
            "batch_size": 8,
            "shuffle": True,
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 5
        },
        "core": {
            "input_size": (61, 121),
            "window_in": 10,  # must be same with batch_gen["window_in_len"]
            "window_out": 5,  # must be same with batch_gen["window_out_len"]
            "num_layers": 1,
            "selected_dim": [5],  # indexes of batch_gen["output_dim"] on batch_gen["input_dim"] list
            "input_attn_params": {
                "input_dim": 10,
                "hidden_dim": 32,
                "attn_channel": 5,
                "kernel_size": 3
            },
            "encoder_params": {
                "input_dim": 9,
                "hidden_dims": [32],
                "kernel_size": [3],
                "bias": False,
                "peephole_con": False
            },
            "decoder_params": {
                "input_dim": 1,  # must be same with len(core["selected_dim"])
                "hidden_dims": [32],
                "kernel_size": [3],
                "bias": False,
                "peephole_con": False
            },
            "output_conv_params": {
                "mid_channel": 5,
                "out_channel": 1,  # must be same with len(batch_gen["output_dim"])
                "in_kernel": 3,
                "out_kernel": 1
            }
        }
    },
    "lstm": {
        "batch_gen": {
            "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],
            "output_dim": 10,
            "window_in_len": 10,
            "window_out_len": 5,
            "batch_size": 8,
            "shuffle": True,
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 6
        },
        "core": {
            "input_size": (61, 121),
            "window_out": 5,  # must be same with batch_gen["window_out_len"]
            "num_layers": 2,
            "selected_dim": 5,
            "hidden_dim": 256,
            "dropout": 0.1,
            "bias": True
        }
    },
    "traj_gru": {
        "batch_gen": {
            "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],
            "output_dim": 10,
            "window_in_len": 10,
            "window_out_len": 5,
            "batch_size": 8,
            "shuffle": True,
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 4
        },
        "core": {
            "input_size": (61, 121),
            "window_in": 10,  # should be same with batch_gen["window_in_len"]
            "window_out": 5,  # should be same with batch_gen["window_out_len"]
            "encoder_params": {
                "input_dim": 9,
                "hidden_dim": 1,
                "kernel_size": 3,
                "bias": False,
                "connection": 1
            },
            "decoder_params": {
                "input_dim": 1,
                "hidden_dim": 1,
                "kernel_size": 3,
                "bias": False,
                "connection": 1
            }
        },
    },
}
