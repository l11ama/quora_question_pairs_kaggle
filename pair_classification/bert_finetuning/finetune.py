"""
Fine-tune BERT model for text classification.

"""

import os
import sys
import time
import ruamel.yaml as yaml
from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, log_loss
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.utils.data

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertAdam

from pair_classification.bert_finetuning.hugginface_pair_clf import BertForSequencePairClassification
from pair_classification.bert_finetuning.util import sigmoid_np


def set_configs(path='config.yml'):
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


# nice way to report running times
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


# make results fully reproducible
def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Converting the lines to BERT format
def convert_lines(example, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"]) + \
                    [0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(f"There are {longer} lines longer than {max_seq_length}")
    return np.array(all_tokens)


def parse_data_to_bert_format(path_to_data, train_file_name,  validate, predict_for_test, test_file_name,
                              text_col_name_1, text_col_name_2, label_col_name, path_to_pretrained_model,
                              max_seq_length, batch_size, toy):
    """

    :param path_to_data:               Path to a folder with CSV files
    :param train_file_name:            ...
    :param validate:                   Whether to run validation with a holdout set
    :param predict_for_test:           Whether to output predictions for the test set
    :param test_file_name:             ...
    :param text_col_name_1:              ...
    :param text_col_name_2:              ...
    :param label_col_name:             ...
    :param path_to_pretrained_model:   Path to a folder with pretrained BERT model
    :param max_seq_length:             ...
    :param batch_size:                 ...
    :param toy:                        Reads first 100 lines of each file for quick test
    :return: train_loader, val_loader, test_loader, class_names - PyTorch loaders and a list of class names.
             Depending on flags `validate` and `predict_for_test`,  val_loader and test_loader might be None.
    """

    path_to_data = Path(path_to_data)

    tokenizer = BertTokenizer.from_pretrained(path_to_pretrained_model, cache_dir=None, do_lower_case=False)
    if tokenizer is None:
        print(f'Please download and unzip your pre-trained model to <{path_to_pretrained_model}>.')
        print('Eg. in case of uncased medium model it can be done by running:')
        print('\twget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')
        sys.exit()
    # We read 5000 lines in a toy test mode or all lines otherwise (nrows=None)
    nrows = 5000 if toy else None
    train_df = pd.read_csv(path_to_data / train_file_name, nrows=nrows)
    print('loaded {} train records'.format(len(train_df)))

    # Make sure all text values are strings
    train_df[text_col_name_1] = train_df[text_col_name_1].astype(str).fillna('DUMMY_VALUE')
    train_df[text_col_name_2] = train_df[text_col_name_2].astype(str).fillna('DUMMY_VALUE')

    X_train_left = convert_lines(train_df[text_col_name_1], max_seq_length, tokenizer)
    X_train_right = convert_lines(train_df[text_col_name_2], max_seq_length, tokenizer)
    X_train = np.stack([X_train_left, X_train_right], axis=-1)

    label_encoder = LabelEncoder().fit(train_df[label_col_name])
    y_train = label_encoder.transform(train_df[label_col_name])

    class_names = label_encoder.classes_
    class_names = np.array(class_names, dtype=str)

    if validate:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long),
                                                   torch.tensor(y_train, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None

    if validate:
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long),
                                                     torch.tensor(y_val, dtype=torch.long))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = None

    if predict_for_test:
        # We read 5000 lines in a toy test mode or all lines otherwise (nrows=None)
        nrows = 5000 if toy else None
        test_df = pd.read_csv(path_to_data / test_file_name, nrows=nrows)

        test_df[text_col_name_1] = test_df[text_col_name_1].astype(str).fillna('DUMMY_VALUE')
        test_df[text_col_name_2] = test_df[text_col_name_2].astype(str).fillna('DUMMY_VALUE')

        X_test_left = convert_lines(test_df[text_col_name_1], max_seq_length, tokenizer)
        X_test_right = convert_lines(test_df[text_col_name_2], max_seq_length, tokenizer)
        X_test = np.stack([X_test_left, X_test_right], axis=-1)

        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names


def setup_bert_model(path_to_pretrained_model, num_classes, epochs, lrate, lrate_clf, batch_size, accum_steps,
                     warmup, apex_mixed_precision, seed, device, train_loader):
    """

    :param path_to_pretrained_model:     Path to a folder with pretrained BERT model
    :param num_classes:                  Number of target classes in the training set
    :param epochs:                       ...
    :param lrate:                        ...
    :param lrate_clf:                        ...
    :param batch_size:                   ...
    :param accum_steps:                  ...
    :param warmup:                       Percent of iterations to perform warmup
    :param apex_mixed_precision:         Whether to use nvidia apex mixed-precision training
    :param seed:                         ...
    :param device:                       ...
    :param train_loader:
    :return: model, optimizer            PyTorch model and optimizer
    """

    path_to_pretrained_model = Path(path_to_pretrained_model)

    convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
        str(path_to_pretrained_model / 'bert_model.ckpt'),
        str(path_to_pretrained_model / 'bert_config.json'),
        str(path_to_pretrained_model / 'pytorch_model.bin')
    )

    seed_everything(seed)

    model = BertForSequencePairClassification.from_pretrained(path_to_pretrained_model,
                                                              cache_dir=None,
                                                              num_labels=1)
    model.zero_grad()

    model = model.to(device)

    return setup_bert_optimizer_for_model(model, epochs, lrate, lrate_clf, batch_size, accum_steps,
                                          warmup, apex_mixed_precision, train_loader)


def setup_bert_optimizer_for_model(model, epochs, lrate, lrate_clf, batch_size, accum_steps,
                                   warmup, apex_mixed_precision, train_loader):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if ('classifier' not in n) and not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if ('classifier' not in n) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if 'classifier' in n],
         'weight_decay': 0.01,  'lr': lrate_clf}
        ]

    num_train_optimization_steps = math.ceil((epochs+1) * len(train_loader) / accum_steps)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=lrate, warmup=warmup,
                         t_total=num_train_optimization_steps)
    if apex_mixed_precision:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    return model, optimizer


def train(model, optimizer, epochs, accum_steps, apex_mixed_precision, output_model_file_name,
          device, train_loader, batch_size, class_names, val_loader=None, bert_freezed=False, weight=None):
    """
    :param model:                       PyTorch model
    :param optimizer:                   PyTorch optimizer
    :param epochs:                      ...
    :param accum_steps:                 ...
    :param apex_mixed_precision:        Whether to use nvidia apex mixed-precision training
    :param output_model_file_name:      Output fine-tuned BERT model name
    :param device:                      ...
    :param train_loader:
    :param val_loader:
    :param batch_size:
    :param class_names:
    """
    loss_history = []
    if weight:
        weight = torch.FloatTensor(weight).to(device)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=weight).to(device)

    tq = tqdm(range(epochs))

    for _ in tq:

        for param in model.parameters():
            param.requires_grad = True

        if bert_freezed:
            for param_name, param in model.named_parameters():
                if 'classifier' not in param_name:
                    param.requires_grad = False

        model = model.train()

        avg_loss = 0.
        lossf = None
        tk0 = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        optimizer.zero_grad()
        for i, (x_batch, y_batch) in tk0:
            y_pred = model(x_batch.to(device),
                           attention_mask=(x_batch > 0).to(device), labels=None)
            loss = loss_func(y_pred, y_batch.type(torch.FloatTensor).to(device))
            if apex_mixed_precision:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (i + 1) % accum_steps == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()

            lossf = 0.9 * lossf + 0.1 * loss.item() if lossf else loss.item()

            avg_loss += loss.item() / len(train_loader)

            loss_history.append(loss.item())

        if val_loader is not None:
            validate(val_loader, model, batch_size, class_names, device)

        tq.set_postfix(avg_loss=avg_loss)

    torch.save(model.state_dict(), output_model_file_name)

    plt.plot(range(len(loss_history)), loss_history)
    plt.ylabel('Loss')
    plt.xlabel('Batch number')
    plt.savefig('loss_plot.png', dpi=300)

    return model


def validate(torch_loader, model, batch_size, class_names, device):
    """
    :param torch_loader:         PyTorch loader object for the validation set
    :param model:                ...
    :param batch_size:           ...
    :param class_names:          Original target class names
    :return: val_pred_probs      NumPy array with predicted probabilities for the validation set
    :param device:               ...
    """

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    val_pred_probs = np.zeros([len(torch_loader.dataset)])

    for i, (x_batch, _) in enumerate(tqdm(torch_loader)):
        pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        val_pred_probs[i * batch_size:(i + 1) * batch_size] = pred.detach().cpu().numpy()

    true_labeles = torch_loader.dataset.tensors[1]
    print("Loss function: ", log_loss(true_labeles, sigmoid_np(val_pred_probs)))

    print("F1-score (micro): ", f1_score(true_labeles, np.array(val_pred_probs >= 0, dtype=np.int), average='micro'))

    print(classification_report(y_true=true_labeles,
                                y_pred=np.array(val_pred_probs >= 0, dtype=np.int),
                                target_names=class_names))
    return val_pred_probs


def predict_for_test(torch_loader, model, batch_size, class_names, test_pred_file_name, device):
    """
    :param torch_loader:         PyTorch loader object for the test set
    :param model:                ...
    :param batch_size:           ...
    :param class_names:          Original target class names
    :param test_pred_file_name:  File to output predictions for the test set
    :param device:               ...
    :return: None
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    test_pred_probs = np.zeros([len(torch_loader.dataset)])

    for i, (x_batch,) in enumerate(tqdm(torch_loader)):
        pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
        test_pred_probs[i * batch_size:(i + 1) * batch_size] = pred.detach().cpu().numpy()

    pd.DataFrame(test_pred_probs).to_csv(test_pred_file_name, index=True, index_label='test_id')

    return test_pred_probs


if __name__ == '__main__':
    config = set_configs()

    device = torch.device(f"cuda:{config['torch_device']}")

    with timer('Read data and convert to BERT format'):
        train_loader, val_loader, test_loader, class_names = parse_data_to_bert_format(
            path_to_data=config['path_to_data'],
            train_file_name=config['train_file_name'],
            text_col_name_1=config['text_col_name_1'],
            text_col_name_2=config['text_col_name_2'],
            label_col_name=config['label_col_name'],
            path_to_pretrained_model=config['path_to_pretrained_model'],
            max_seq_length=config['max_seq_length'],
            batch_size=config['batch_size'],
            validate=config['validate'],
            predict_for_test=config['predict_for_test'],
            test_file_name=config['test_file_name'],
            toy=config['toy']
        )

    with timer('Setting up BERT model'):
        model, optimizer = setup_bert_model(
            path_to_pretrained_model=config['path_to_pretrained_model'],
            num_classes=len(class_names), epochs=config['epochs'],
            lrate=config['lrate'],
            lrate_clf=config['lrate_clf'],
            batch_size=config['batch_size'],
            accum_steps=config['accum_steps'], warmup=config['warmup'],
            apex_mixed_precision=config['apex_mixed_precision'],
            seed=config['seed'], device=device, train_loader=train_loader
        )

    with timer('Training'):
        model = train(
            model=model,
            optimizer=optimizer,
            epochs=config['epochs'],
            accum_steps=config['accum_steps'],
            apex_mixed_precision=config['apex_mixed_precision'],
            output_model_file_name=config['output_model_file_name'],
            device=device,
            train_loader=train_loader,
            batch_size=config['batch_size'],
            class_names=class_names
        )

    if config['validate']:
        with timer('Validating with a holdout set'):
            validate(
                torch_loader=val_loader,
                model=model,
                batch_size=config['batch_size'],
                class_names=class_names,
                device=device
            )

    if config['predict_for_test']:
        with timer('Predicting for test set'):
            predict_for_test(
                torch_loader=test_loader,
                model=model,
                batch_size=config['batch_size'],
                test_pred_file_name=config['test_pred_file_name'],
                class_names=class_names,
                device=device
            )
