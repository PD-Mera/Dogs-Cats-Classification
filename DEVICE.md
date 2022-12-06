# HOW TO USE ON EACH DEVICE

## Python 3.10.6 + CUDA 11.8 (RTX 3060 6GB)

``` bash
python3 -m pip install --upgrade pip --no-cache-dir

pip install -r requirements.txt
```

## Python 3.6.9 + CUDA 11.1 (RTX 3060 12GB Server)

``` bash
python3 -m pip install --upgrade pip --no-cache-dir

pip3 install torch==1.9.0+cu111 \
             torchvision==0.10.0+cu111 \
             torchaudio==0.9.0 \
             -f https://download.pytorch.org/whl/torch_stable.html
```

In `model.py`:

``` python
# Change
backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
# To
backbone = resnet18(pretrained=True)
```

In `train.py`

``` python
# Change
loss = loss_fn(output, target)
# To
loss = loss_fn(output, torch.max(target, 1)[1])
```