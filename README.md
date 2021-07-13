## ssd300-pytorch-onnx-tensorrt-benchmark

### Setup pytorch environment
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```
### Run benchmark with ssd300 model on pytorch, onnx and tensorrt platform
```
python main.py -i media/image_1.jpg
```
