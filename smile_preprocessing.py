import pandas as pd


def smile_data_clean():
    df = pd.read_csv('smile-annotations.csv',
                    names=['id', 'text', 'category'])
    df.set_index('id', inplace=True)

    df.category.value_counts()

    df = df[~df.category.str.contains('\|')]
    df = df[df.category != 'nocode']

    df.category.value_counts()
    return df