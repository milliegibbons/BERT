import pandas as pd


def smile_data_clean():
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
    return df