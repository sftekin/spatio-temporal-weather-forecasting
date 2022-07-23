from configs.config_generator import Param
from configs.config import model_params

# Order of features in the input tensors are:
# ['q', 'tcc', 'tisr', 'u10', 'v10', 'z', 'pv', 'r', 't2m', 'tp', 'u', 'v',
# 'vo', 't', 'lat', 'lon', 'orography', 'lsm', 'slt', 'lat2d', 'lon2d']
# where "t" is the temperature and at the 13th index


# WEATHER MODEL
model_params["weather_model"]["batch_gen"]["input_dim"] = "all"
model_params["weather_model"]["batch_gen"]["output_dim"] = [13]
model_params["weather_model"]["batch_gen"]["window_in_len"] = 10
model_params["weather_model"]["batch_gen"]["window_out_len"] = 3
model_params["weather_model"]["batch_gen"]["batch_size"] = 32
model_params["weather_model"]["batch_gen"]["temporal_freq"] = 24
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
