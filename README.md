# SpaceNet7 Challenge

train on cpu:

```
python run.py trainer.gpus=0
```

train on gpu:

```
python run.py trainer.gpus=1
```

predict:

```
python run.py mode=predict
```

building the image for train:
```
docker run --gpus '"device=0"' --ipc=host --rm --user $(id -u):$(id -g) --name run_code -v "$PWD":/workspace/project -t spacenet python run.py
```

building the image for predict:
```
docker run --gpus '"device=0"' --ipc=host --rm --user $(id -u):$(id -g) --name run_code -v "$PWD":/workspace/project -it spacenet python run.py mode=predic
```
