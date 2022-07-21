from experimenter import train_test, get_exp_count
from inference import inference_on_test


def run_weatherbenc(model_name, exp_type):
    from configs.weatherbench.default_conf import experiment_params, data_params
    if exp_type == "iterative":
        from configs.weatherbench.iter_model_confs import model_params
    elif exp_type == "sequential":
        from configs.weatherbench.seq_model_confs import model_params
    else:
        raise KeyError("exp_type can only be either 'itereative' or 'sequential'")

    # perform train test
    train_test(experiment_params=experiment_params,
               data_params=data_params,
               model_params=model_params)

    # perform inference on test
    inference_params = {
        "model_name": model_name,
        "start_date_str": "01-01-2017",
        "end_date_str": "01-01-2018",
        "test_data_folder": "data/weatherbench/test_data",
        "exp_num": get_exp_count(model_name),  # get the last experiment
        # "exp_num": 2,  # or set it by yourself
        "forecast_horizon": 3,
        "selected_dim": -1  # index position of the selected feature
    }
    inference_on_test(device=experiment_params["device"], **inference_params)


def run_highres():
    from configs.higher_res.higher_res_config import experiment_params, data_params, model_params

    # perform train test
    train_test(experiment_params=experiment_params,
               data_params=data_params,
               model_params=model_params)


if __name__ == '__main__':
    # perform experiments on weatherbench
    run_weatherbenc(model_name="weather_model", exp_type="sequential")

    # # perform experiments on highres
    # run_highres()
