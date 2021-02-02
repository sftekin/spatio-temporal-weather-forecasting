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

Will be updated...
