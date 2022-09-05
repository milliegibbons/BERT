import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import random
from result_functions import f1_score_func, accuracy_per_class, accuracy, incorrect



def create_model(df):

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

    df['data_type'] = ['not_set'] * df.shape[0]

    df.loc[x_train, 'data_type'] = 'train'
    df.loc[x_val, 'data_type'] = 'val'

    df.groupby(['category', 'label', 'data_type']).count()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == 'train'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt')

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == 'val'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt')

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type == 'train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type == 'val'].label.values)

    dataset_train = TensorDataset(input_ids_train,
                                  attention_masks_train,
                                  labels_train)

    dataset_val = TensorDataset(input_ids_val,
                                attention_masks_val,
                                labels_val)

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_dict),
        output_attentions=True,
        output_hidden_states=False)

    batch_size = 4
    data_loader_train = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=batch_size)

    data_loader_val = DataLoader(
        dataset_val,
        sampler=RandomSampler(dataset_val),
        batch_size=32)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-5,
        eps=1e-8)

    epochs = 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(data_loader_train) * epochs)

    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    # torch.cuda.manual_seed_all(seed_val)

    device = torch.device('cpu')
    model.to(device)
    print(device)

    def evaluate(data_loader_val):
        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in tqdm(data_loader_val):
            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(data_loader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals

    for epoch in tqdm(range(1, epochs + 1)):

        model.train()

        loss_train_total = 0

        progress_bar = tqdm(data_loader_train,
                            desc='Epoch {:1d}'.format(epoch),
                            leave=False,
                            disable=False)

        for batch in progress_bar:
            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

        torch.save(model.state_dict(), f'epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(data_loader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(data_loader_val)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                              num_labels=len(label_dict),
                                                              output_attentions=False,
                                                              output_hidden_states=False)

        model.to(device)

        model.load_state_dict(torch.load('epoch_1.model', map_location=torch.device('cpu')))

        _, predictions, true_vals = evaluate(data_loader_val)

        per_class = accuracy_per_class(predictions, true_vals, label_dict)

        accuracy_df = accuracy(predictions, true_vals, df)

        incorrect_df = incorrect(predictions, true_vals, label_dict, df)











