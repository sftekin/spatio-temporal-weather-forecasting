from experimenter import train_test, get_exp_count
from inference import inference_on_test


def run_weatherbenc():
    from configs.weatherbench_config import experiment_params, data_params, model_params

    # perform train test
    train_test(experiment_params=experiment_params,
               data_params=data_params,
               model_params=model_params)

    # perform inference on test
    model_name = "weather_model"
    inference_params = {
        "model_name": model_name,
        "start_date_str": "01-01-2017",
        "end_date_str": "01-01-2018",
        "test_data_folder": "data/weatherbench/test_data",
        "exp_num": get_exp_count(model_name),  # get the last experiment
        # "exp_num": 1  # or set it by yourself
        "forecast_horizon": 72,
        "selected_dim": -1  # index position of the selected feature
    }
    inference_on_test(device=experiment_params["device"], **inference_params)


def run_highres():
    from configs.higher_res_config import experiment_params, data_params, model_params

    # perform train test
    train_test(experiment_params=experiment_params,
               data_params=data_params,
               model_params=model_params)


if __name__ == '__main__':
    # perform experiments on weatherbench
    run_weatherbenc()

    # # perform experiments on highres
    # run_highres()
