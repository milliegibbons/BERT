import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels, label_dict):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]

        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds] == label)}/{len(y_true)}\n')



def accuracy(preds, labels, df):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    rows = []
    for idx in range(len(preds_flat)):
        row = [df[df.data_type == 'val'].Data.values[idx], preds_flat[idx], labels_flat[idx]]
        rows.append(row)
    return pd.DataFrame(rows)


def incorrect(preds, labels, label_dict, df):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    rows = []
    for idx in range(len(preds_flat)):
        if preds_flat[idx] != labels_flat[idx]:
            row = [df[df.data_type == 'val'].Data.values[idx], preds_flat[idx], labels_flat[idx]]
            rows.append(row)
    return pd.DataFrame(rows)