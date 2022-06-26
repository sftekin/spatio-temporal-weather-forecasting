from config import experiment_params, data_params, model_params
from experimenter import train_test, inference_on_test


def run(experiment_type="train_test", exp_num=None, model_name=None):
    if experiment_type == "train_test":
        train_test(experiment_params=experiment_params,
                   data_params=data_params,
                   model_params=model_params)
    elif experiment_type == "inference":
        inference_on_test(model_name=model_name,
                          exp_num=exp_num,
                          device=experiment_params["device"])
    else:
        raise KeyError("Wrong experiment_type, it can be either 'train_tester' or 'inference'")


if __name__ == '__main__':
    # perform train test
    run(experiment_type="train_test")

    # # perform inference on test
    # model_name = "weather_model"
    # exp_number = get_exp_count(model_name)  # get the last experiment
    # # exp_number = 10  # or set by yourself
    # run(experiment_type="inference", exp_num=exp_number, model_name="weather_model")
