from config_generator import Param
from configs.config import experiment_params, data_params, model_params

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

# overwrite model parameters
model_names = ["moving_avg", "convlstm", "u_net", "weather_model", "lstm", "traj_gru"]
for model_n in model_names:
    model_params[model_n]["batch_gen"]["input_dim"] = [0, 7, 8, 13]
    model_params[model_n]["batch_gen"]["output_dim"] = [0, 7, 8, 13]
    model_params[model_n]["batch_gen"]["window_in_len"] = 6
    model_params[model_n]["batch_gen"]["window_out_len"] = 6
    model_params[model_n]["batch_gen"]["batch_size"] = 32
    model_params[model_n]["trainer"]["num_epochs"] = 50

model_params["weather_model"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005])
model_params["u_net"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["convlstm"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["moving_avg"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["lstm"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["traj_gru"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])

model_params["convlstm"]["core"]["input_size"] = (32, 64)
model_params["weather_model"]["core"]["input_size"] = (32, 64)
model_params["lstm"]["core"]["input_size"] = (32, 64)
model_params["traj_gru"]["core"]["input_size"] = (32, 64)

model_params["u_net"]["core"]["selected_dim"] = 13  # index of temperature
model_params["weather_model"]["core"]["selected_dim"] = [0, 1, 2, 3]
model_params["weather_model"]["trainer"]["selected_dim"] = -1  # index of temperature
model_params["convlstm"]["trainer"]["selected_dim"] = -1  # index of temperature
model_params["lstm"]["core"]["selected_dim"] = 13

model_params["weather_model"]["core"]["encoder_params"]["input_dim"] = \
    len(model_params["weather_model"]["batch_gen"]["input_dim"])
model_params["convlstm"]["core"]["encoder_params"]["input_dim"] = \
    len(model_params["convlstm"]["batch_gen"]["input_dim"])
model_params["traj_gru"]["core"]["encoder_params"]["input_dim"] = 19

model_names.remove("u_net")
for model_n in model_names:
    model_params[model_n]["core"]["window_in"] = model_params[model_n]["batch_gen"]["window_in_len"]
    model_params[model_n]["core"]["window_out"] = model_params[model_n]["batch_gen"]["window_out_len"]

model_params["u_net"]["core"]["in_channels"] = model_params["u_net"]["batch_gen"]["window_in_len"]
model_params["u_net"]["core"]["out_channels"] = model_params["u_net"]["batch_gen"]["window_out_len"]
model_params["weather_model"]["core"]["input_attn_params"]["input_dim"] = \
    model_params["weather_model"]["batch_gen"]["window_in_len"]

model_params["weather_model"]["core"]["output_conv_params"]["out_channel"] = \
    len(model_params["weather_model"]["batch_gen"]["output_dim"])
model_params["weather_model"]["core"]["decoder_params"]["input_dim"] = \
    len(model_params["weather_model"]["core"]["selected_dim"])

model_params["convlstm"]["core"]["encoder_params"]["hidden_dims"] = \
    [len(model_params["convlstm"]["batch_gen"]["input_dim"])]
model_params["convlstm"]["core"]["decoder_params"]["hidden_dims"] = \
    [len(model_params["convlstm"]["batch_gen"]["output_dim"])]
