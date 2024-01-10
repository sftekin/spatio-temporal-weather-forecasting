from experimenter import train_test, get_exp_count
from inference import inference_on_test


def run_weatherbenc(model_name, exp_type, perform_training=True):
    from configs.weatherbench.default_conf import experiment_params, data_params
    if exp_type == "iterative":
        from configs.weatherbench.iter_model_confs import model_params
    elif exp_type == "sequential":
        from configs.weatherbench.seq_model_confs import model_params
    elif exp_type == "direct":
        from configs.weatherbench.direct_model_confs import model_params
    else:
        raise KeyError("exp_type can only be either 'itereative' or 'sequential'")

    # perform train test
    if perform_training:
        train_test(experiment_params=experiment_params,
                   data_params=data_params,
                   model_params=model_params,
                   exp_type=exp_type)

    # perform inference on test
    exp_dir = f"results/weatherbench/{exp_type}_results"
    inference_params = {
        "model_name": model_name,
        "start_date_str": "01-01-2017",
        "end_date_str": "01-01-2018",
        "test_data_folder": "data/weatherbench/test_data",
        "exp_dir": exp_dir,
        "exp_num": get_exp_count(model_name, result_dir=exp_dir),  # get the last experiment
        # "exp_num": 2,  # or set it by yourself
        "forecast_horizon": 72,
        "selected_dim": -1  # index position of the selected feature
    }
    inference_on_test(dataset_type="weatherbench", device=experiment_params["device"], **inference_params)


def run_highres(model_name, perform_training=True):
    from configs.higher_res.higher_res_config import experiment_params, data_params, model_params

    # perform train test
    if perform_training:
        train_test(experiment_params=experiment_params,
                   data_params=data_params,
                   model_params=model_params)

    # perform inference on test
    exp_dir = f"results/highres"
    inference_params = {
        "model_name": model_name,
        "start_date_str": "01-01-2001",
        "end_date_str": "02-01-2001",
        "test_data_folder": "data/data_dump",
        "exp_dir": exp_dir,
        "exp_num": get_exp_count(model_name, result_dir=exp_dir),  # get the last experiment
        # "exp_num": 2,  # or set it by yourself
        "forecast_horizon": 5,
        "selected_dim": -1  # index position of the selected feature
    }
    inference_on_test(dataset_type="highres", device=experiment_params["device"], **inference_params)


def run(dataset, model_name, perform_training, exp_type=None):
    if dataset == "weatherbench":
        # perform experiments on weatherbench
        run_weatherbenc(model_name=model_name, exp_type=exp_type, perform_training=perform_training)
    elif dataset == "highres":
        # perform experiments on highres
        run_highres(model_name="weather_model", perform_training=perform_training)
    else:
        print("The dataset can only be either 'weatherbench' or 'highres'")


if __name__ == '__main__':
    run(dataset="highres",
        model_name="weather_model",
        exp_type="",
        perform_training=True)

