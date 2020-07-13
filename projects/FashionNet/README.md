Please follow the [INSTALL.md](../../INSTALL.md) to install the detectron2 first

## Quick Start
1. Run the inference 

```shell script
cd projects/FashionNet
python demo.py --config-file configs/fashionnet_R_50_FPN.yaml \
    --input [your_input_image] \  
    --output [your_result_dir] \
    --opts MODEL.WEIGHTS [your_model.pth]
```
2. Train the model 
```shell script
cd projects/FashionNet
export DETECTRON2_DATASETS=/your/datasets/path
python train_net.py --num-gpus 2 \
    --config-file configs/fashionnet_R_50_FPN.yaml \
    --dist-url auto
```
3. Evaluation
```shell script
cd projects/FashionNet
python train_net.py --num-gpus 2 \
    --config-file configs/fashionnet_R_50_FPN.yaml \
    --dist-url auto --eval-only
```

4. Export Model to Onnx 
```python
python onnx_converter.py \
    --config-file configs/fashionnet_R_50_FPN.yaml \
    --output ./output
```