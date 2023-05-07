# About

Encoder-decoder tranformer model implemented using https://github.com/karpathy/nanoGPT as a base.

# Prerequisites

- Pytorch
- pip install transformers
- pip install datasets
- pip install wandb
- pip install tqdm 

# How to run
### Prepare data
Run python data/{dataset name}/prepare.py in the cmd to download a huggingface dataset and tokenize + split it

### Train the model
- Run python train.py config/{config name}.py
- Model parameters can be overwritten in the cmd by typing python train.py config/{config name}.py --{parameter}={value}

# Inference
Run python sample.py --out_dir={folder containing model checkpoint} --prompt="Example prompt"
