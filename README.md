# Dogs and Cats Classification

This repo build a classification pipeline for dogs and cats classification

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

- Download Dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats) or download processed Dataset from [here](https://drive.google.com/file/d/1hI2s-U7PwlQUFc8eHtBtl1_TCLtDliId/view?usp=share_link)

- Data in this format

``` files
|-- data
    |-- train
    |   |-- class 1
    |   |-- class 2
    |   `-- ...
    |-- valid
    |   |-- class 1
    |   |-- class 2
    |   `-- ...
    `-- test
        |-- class 1
        |-- class 2
        `-- ...
```

### Config

Modify infomation about training in `config.py`

### Train

Simply run 

``` bash
python train.py
```

### Experiment Results

Results after training 10 epochs

| Model       | Training Info | Best Accuracy |
| ----------- |:-------------:| :-----------: |
| Resnet18    | Adam, lr=1e-5 |               |
| Resnet50    |               |               |


## TODO

- [x] Split Dataset to Train and Valid folder
- [x] Building a pipeline to training Dogs and Cats Classification Model
    - [x] Building Model
    - [x] Building Dataloader
    - [x] Building Training Pipeline
    - [x] Building Testing Pipeline
- [ ] Training Dogs and Cats Classification Model
- [ ] Expanding pipeline for Classification Tasks