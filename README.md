# Image Classification Task

This repo build a classification pipeline and example for dogs and cats classification

## Dependencies and Installation

### Environments

- Python 3.10.6 + CUDA 11.8

Refer to [DEVICE.md](./DEVICE.md) for more environments

### Install requirements

``` bash
pip install -r requirements.txt
```

## Prepare

### Data

- Download Dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats) or download processed Dataset from [here]()

### Config

Modify infomation about training in `config.py`

## TODO

- [x] Split Dataset to Train and Valid folder
- [x] Building a pipeline to training Dogs and Cats Classification Model
    - [x] Building Model
    - [x] Building Dataloader
    - [x] Building Training Pipeline
    - [x] Building Testing Pipeline
- [ ] Training Dogs and Cats Classification Model
- [ ] Expanding pipeline for Classification Tasks