import torch
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import f1_score
import random

df = pd.read_csv('smile-annotations-final.csv',
                names=['id', 'text', 'category'])
df.set_index('id', inplace=True)

df.category.value_counts()

df = df[~df.category.str.contains('\|')]
df = df[df.category != 'nocode']

df.category.value_counts()

possible_labels = df.category.unique()

label_dict = {}
for idx, label in enumerate(possible_labels):
    label_dict[label] = idx

df['label'] = df.category.replace(label_dict)

x_train, x_val, y_train, y_val = train_test_split(df.index.values,
                                                 df.label.values,
                                                 test_size=0.15,
                                                 random_state=42,
                                                 stratify=df.label.values)

df['data_type'] = ['not_set']*df.shape[0]

df.loc[x_train, 'data_type'] = 'train'
df.loc[x_val, 'data_type'] = 'val'

df.groupby(['category', 'label', 'data_type']).count()