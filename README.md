# AlexNet
Implementation of well-known ['AlexNet' architecture](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) using PyTorch. Currently using the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

## Usage
Ensure your directory is laid out as shown:
```
├── data/
├── src
│    ├── data.py
│    ├── eval.py
│    ├── model.py
│    └── train.py
└── weights/
```
Make sure to install all requirements using: `pip install -r requirements.txt`

### Training a model
Run `python3 train.py`. Ensure to change the number of epochs you would like to train for.

### Evaluating a model
Currently, the evaluation simply calculates the top-1 accuracy over the test datasplit. This can be done by running `python3 eval.py`.
