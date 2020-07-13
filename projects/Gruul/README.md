## Train Model
```shell script
cd projects/Gruul
export DETECTRON2_DATASETS=/your/datasets/path
python train_net.py --num-gpus 2 \
    --config-file configs/R_50_2x.yaml \
    --dist-url auto
```

## Eval Model
```shell script
cd projects/Gruul
export DETECTRON2_DATASETS=/your/datasets/path
python train_net.py --num-gpus 2 \
    --config-file configs/R_50_2x.yaml \
    --dist-url auto \
    --eval-only
```