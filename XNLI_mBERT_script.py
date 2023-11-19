import argparse
import os
import random
import torch
import pandas as pd
import sklearn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
from transformers import AutoModelForMultipleChoice
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate

def get_datas(dataset):
    dataset = load_dataset(dataset,"ARC-Challenge")
    train_dataset = dataset['train'].filter(filter_func)
    test_dataset = dataset['test'].filter(filter_func)
    val_dataset = dataset['validation'].filter(filter_func)
    
    return train_dataset,test_dataset,val_dataset

dataset = "xnli"