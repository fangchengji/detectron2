## Quick Start
1. Run the inference 

```
cd projects/FashionNet
python demo.py --config-file configs/fashionnet_R_50.yaml \
    --input [your_input_image] \  
    --output [your_result_dir] \
    --opts MODEL.WEIGHTS [your_model.pth]
```
2. Train the model 
```
cd projects/FashionNet
python train_net.py --num-gpus 2 \
    --config-file configs/fashionnet_R_50.yaml \
    --dist-url auto
```
3. Evaluation
```
cd projects/FashionNet
python train_net.py --num-gpus 1 \
    --config-file configs/fashionnet_R_50.yaml \
    --dist-url auto --eval-only
```