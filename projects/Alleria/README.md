# Train
```python
python train.py --num-gpus 2 --config-file configs/atss_R_50_FPN_1x.yaml --dist-url auto
```

# Eval
```python
python train.py --num-gpus 4 --config-file configs/qfl_atss_ensemble.yaml --dist-url auto --eval-only
```

# Inference
```python
python demo.py --config-file configs/qfl_atss_ensemble.yaml \
    --input /data/fangcheng.ji/datasets/wheat/pseudo_test/*.jpg \
    --output /data/fangcheng.ji/datasets/wheat/pseudo_test_out 
```