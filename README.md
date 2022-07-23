# The Weather Model

A spatio-temporal forecasting model for the Numerical Weather Prediction.

## Install & Run

```
$ docker build -t weather:v1 docker/.
```

```
$ docker run -dit --name weather_model --gpus all --rm weather:v1 python run.py 
```

```
$ docker logs -f weather_model
```

## Paper Link
[http://arxiv.org/abs/2102.00696](http://arxiv.org/abs/2102.00696)

## Dataset Link

https://drive.google.com/drive/folders/1Ry1x-Fv6uxtLYfgRBv-MUOLaWjK3wPO6?usp=sharing

download the highres and weatherbench datasets and put it under the data directory. The locations should be:

highres --> data/data_dump
weatherbench --> data/weatherbench
