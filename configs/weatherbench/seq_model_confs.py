from config_generator import Param
from configs.config import model_params

# Order of features in the input tensors are:
# ['q', 'tcc', 'tisr', 'u10', 'v10', 'z', 'pv', 'r', 't2m', 'tp', 'u', 'v',
# 'vo', 't', 'lat', 'lon', 'orography', 'lsm', 'slt', 'lat2d', 'lon2d']
# where "t" is the temperature and at the 13th index


# WEATHER MODEL
model_params["weather_model"]["batch_gen"]["input_dim"] = "all"
model_params["weather_model"]["batch_gen"]["output_dim"] = [13]
model_params["weather_model"]["batch_gen"]["window_in_len"] = 20
model_params["weather_model"]["batch_gen"]["window_out_len"] = 72
model_params["weather_model"]["batch_gen"]["batch_size"] = 8
model_params["weather_model"]["batch_gen"]["temporal_freq"] = 1
model_params["weather_model"]["batch_gen"]["max_temporal_freq"] = 1
model_params["weather_model"]["trainer"]["num_epochs"] = 50
model_params["weather_model"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005])
model_params["weather_model"]["trainer"]["selected_dim"] = -1  # index of temperature
model_params["weather_model"]["core"]["input_size"] = (32, 64)
model_params["weather_model"]["core"]["selected_dim"] = [13]
model_params["weather_model"]["core"]["window_in"] = model_params["weather_model"]["batch_gen"]["window_in_len"]
model_params["weather_model"]["core"]["window_out"] = model_params["weather_model"]["batch_gen"]["window_out_len"]
model_params["weather_model"]["core"]["encoder_params"]["input_dim"] = 19
model_params["weather_model"]["core"]["input_attn_params"]["input_dim"] = model_params["weather_model"]["batch_gen"]["window_in_len"]
model_params["weather_model"]["core"]["output_conv_params"]["out_channel"] = len(model_params["weather_model"]["batch_gen"]["output_dim"])
model_params["weather_model"]["core"]["decoder_params"]["input_dim"] = len(model_params["weather_model"]["core"]["selected_dim"])

# CONVLSTM
model_params["convlstm"]["batch_gen"]["input_dim"] = "all"
model_params["convlstm"]["batch_gen"]["output_dim"] = [13]
model_params["convlstm"]["batch_gen"]["window_in_len"] = 20
model_params["convlstm"]["batch_gen"]["window_out_len"] = 72
model_params["convlstm"]["batch_gen"]["batch_size"] = 8
model_params["convlstm"]["batch_gen"]["temporal_freq"] = 1
model_params["convlstm"]["batch_gen"]["max_temporal_freq"] = 1
model_params["convlstm"]["trainer"]["num_epochs"] = 50
model_params["convlstm"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["convlstm"]["trainer"]["selected_dim"] = -1  # index of temperature
model_params["convlstm"]["core"]["input_size"] = (32, 64)
model_params["convlstm"]["core"]["selected_dim"] = [13]
model_params["convlstm"]["core"]["window_in"] = model_params["convlstm"]["batch_gen"]["window_in_len"]
model_params["convlstm"]["core"]["window_out"] = model_params["convlstm"]["batch_gen"]["window_out_len"]
model_params["convlstm"]["core"]["encoder_params"]["input_dim"] = 19
model_params["convlstm"]["core"]["encoder_params"]["hidden_dims"] = [len(model_params["convlstm"]["batch_gen"]["input_dim"])]
model_params["convlstm"]["core"]["decoder_params"]["hidden_dims"] = [len(model_params["convlstm"]["batch_gen"]["output_dim"])]
model_params["convlstm"]["core"]["decoder_params"]["input_dim"] = len(model_params["convlstm"]["core"]["selected_dim"])

# U_NET
model_params["u_net"]["batch_gen"]["input_dim"] = "all"
model_params["u_net"]["batch_gen"]["output_dim"] = [13]
model_params["u_net"]["batch_gen"]["window_in_len"] = 20
model_params["u_net"]["batch_gen"]["window_out_len"] = 72
model_params["u_net"]["batch_gen"]["batch_size"] = 8
model_params["u_net"]["trainer"]["num_epochs"] = 50
model_params["u_net"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["u_net"]["core"]["selected_dim"] = 13  # index of temperature
model_params["u_net"]["core"]["in_channels"] = model_params["u_net"]["batch_gen"]["window_in_len"]
model_params["u_net"]["core"]["out_channels"] = model_params["u_net"]["batch_gen"]["window_out_len"]

# TRAJ_GRU
model_params["traj_gru"]["batch_gen"]["input_dim"] = "all"
model_params["traj_gru"]["batch_gen"]["output_dim"] = [13]
model_params["traj_gru"]["batch_gen"]["window_in_len"] = 20
model_params["traj_gru"]["batch_gen"]["window_out_len"] = 72
model_params["traj_gru"]["batch_gen"]["batch_size"] = 8
model_params["traj_gru"]["batch_gen"]["temporal_freq"] = 1
model_params["traj_gru"]["batch_gen"]["max_temporal_freq"] = 1
model_params["traj_gru"]["trainer"]["num_epochs"] = 50
model_params["traj_gru"]["trainer"]["learning_rate"] = Param([0.01, 0.001, 0.0005, 0.00001])
model_params["traj_gru"]["trainer"]["selected_dim"] = -1  # index of temperature
model_params["traj_gru"]["core"]["input_size"] = (32, 64)
model_params["traj_gru"]["core"]["window_in"] = model_params["traj_gru"]["batch_gen"]["window_in_len"]
model_params["traj_gru"]["core"]["window_out"] = model_params["traj_gru"]["batch_gen"]["window_out_len"]
model_params["traj_gru"]["core"]["encoder_params"]["input_dim"] = 19
model_params["traj_gru"]["core"]["encoder_params"]["hidden_dim"] = 1
model_params["traj_gru"]["core"]["decoder_params"]["hidden_dim"] = 1
model_params["traj_gru"]["core"]["decoder_params"]["input_dim"] = 1


