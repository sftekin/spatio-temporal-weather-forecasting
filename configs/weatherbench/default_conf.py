from configs.config import experiment_params, data_params

# overwrite experiment parameters
experiment_params["global_start_date"] = '2015-01-01'
experiment_params["global_end_date"] = '2016-12-31'
experiment_params["data_step"] = 6  # in month
experiment_params["data_length"] = 24  # in month
experiment_params["val_ratio"] = 0.1
experiment_params["test_ratio"] = 0.0
experiment_params["normalize_flag"] = True
experiment_params["device"] = "cuda"
experiment_params["model"] = "weather_model"

# overwrite data parameters
data_params["rebuild"] = False
data_params["dump_data_folder"] = "weatherbench/train_data"
