from configs.config_generator import Param
from configs.config import experiment_params, data_params, model_params

# overwrite experiment parameters
experiment_params["global_start_date"] = '2000-01-01'
experiment_params["global_end_date"] = '2008-12-31'
experiment_params["data_step"] = 6  # in month
experiment_params["data_length"] = 12  # in month
experiment_params["val_ratio"] = 0.1
experiment_params["test_ratio"] = 0.1
experiment_params["normalize_flag"] = True
experiment_params["device"] = "cuda"
experiment_params["model"] = "weather_model"

# overwrite data parameters
data_params["rebuild"] = False
data_params["dump_data_folder"] = "data_dump"

# Order of features in the input tensors are:
# ['d', 'cc', 'z', 'pv', 'r', 'ciwc', 'clwc', 'q', 'crwc', 'cswc', 't', 'u', 'v', 'w']
# where "t" is the temperature and at the 13th index

# overwrite model parameters
model_names = ["moving_avg", "convlstm", "u_net", "weather_model", "lstm", "traj_gru"]
for model_n in model_names:
    model_params[model_n]["batch_gen"]["input_dim"] = [0, 2, 3, 4, 7, 10, 11, 12, 13]
    model_params[model_n]["batch_gen"]["output_dim"] = 10
    model_params[model_n]["batch_gen"]["window_in_len"] = 10
    model_params[model_n]["batch_gen"]["window_out_len"] = 5
    model_params[model_n]["batch_gen"]["batch_size"] = 8
    model_params[model_n]["batch_gen"]["temporal_freq"] = 1
    model_params[model_n]["batch_gen"]["max_temporal_freq"] = 1
    model_params[model_n]["trainer"]["num_epochs"] = 50

model_params["weather_model"]["trainer"]["learning_rate"] = Param([0.001, 0.0005])
model_params["u_net"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["convlstm"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["moving_avg"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["lstm"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["traj_gru"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])

model_params["convlstm"]["core"]["input_size"] = (61, 121)
model_params["weather_model"]["core"]["input_size"] = (61, 121)
model_params["lstm"]["core"]["input_size"] = (61, 121)
model_params["traj_gru"]["core"]["input_size"] = (61, 121)

model_params["u_net"]["core"]["selected_dim"] = 5
model_params["weather_model"]["core"]["selected_dim"] = [5]
model_params["lstm"]["core"]["selected_dim"] = 5

model_params["weather_model"]["core"]["encoder_params"]["input_dim"] = 9
model_params["convlstm"]["core"]["encoder_params"]["input_dim"] = 9
model_params["traj_gru"]["core"]["encoder_params"]["input_dim"] = 9

model_names.remove("u_net")
for model_n in model_names:
    model_params[model_n]["core"]["window_in"] = model_params[model_n]["batch_gen"]["window_in_len"]
    model_params[model_n]["core"]["window_out"] = model_params[model_n]["batch_gen"]["window_out_len"]

model_params["u_net"]["core"]["in_channels"] = model_params["u_net"]["batch_gen"]["window_in_len"]
model_params["u_net"]["core"]["out_channels"] = model_params["u_net"]["batch_gen"]["window_out_len"]
model_params["weather_model"]["core"]["input_attn_params"]["input_dim"] = \
    model_params["weather_model"]["batch_gen"]["window_in_len"]
