"""
Fine-tune BERT model for text classification.

"""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.utils.data

from pytorch_pretrained_bert import BertTokenizer, BertAdam, convert_tf_checkpoint_to_pytorch

from pair_classification.bert_finetuning.finetune import set_configs, setup_bert_model
from pair_classification.bert_finetuning.hugginface_pair_clf import BertForSequencePairClassificationTriplets
from pair_classification.bert_finetuning.losses import TripletLoss
from pair_classification.bert_finetuning.util import sigmoid_np, tqdm_ext, convert_lines, seed_everything, timer


def parse_data_to_bert_format(path_to_data, train_file_name, anchor_col_name, pos_col_name, neg_col_name,
                                  path_to_pretrained_model, max_seq_length, batch_size, toy):

    path_to_data = Path(path_to_data)

    tokenizer = BertTokenizer.from_pretrained(path_to_pretrained_model, cache_dir=None, do_lower_case=False)
    if tokenizer is None:
        print(f'Please download and unzip your pre-trained model to <{path_to_pretrained_model}>.')
        print('Eg. in case of uncased medium model it can be done by running:')
        print('\twget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')
        sys.exit()

    nrows = 5000 if toy else None
    train_df = pd.read_csv(path_to_data / train_file_name, nrows=nrows)
    print('loaded {} train records'.format(len(train_df)))

    # Make sure all text values are strings
    train_df[anchor_col_name] = train_df[anchor_col_name].astype(str).fillna('DUMMY_VALUE')
    train_df[pos_col_name] = train_df[pos_col_name].astype(str).fillna('DUMMY_VALUE')
    train_df[neg_col_name] = train_df[neg_col_name].astype(str).fillna('DUMMY_VALUE')

    X_train_a = convert_lines(train_df[anchor_col_name], max_seq_length, tokenizer)
    X_train_p = convert_lines(train_df[pos_col_name], max_seq_length, tokenizer)
    X_train_n = convert_lines(train_df[neg_col_name], max_seq_length, tokenizer)
    X_train = np.stack([X_train_a, X_train_p, X_train_n], axis=-1)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def train(model, optimizer, epochs, accum_steps, apex_mixed_precision, output_model_file_name,
          device, train_loader, bert_freezed=False):
    """
    :param model:                       PyTorch model
    :param optimizer:                   PyTorch optimizer
    :param epochs:                      ...
    :param accum_steps:                 ...
    :param apex_mixed_precision:        Whether to use nvidia apex mixed-precision training
    :param output_model_file_name:      Output fine-tuned BERT model name
    :param device:                      ...
    :param train_loader:
    """
    loss_history = []
    loss_func = TripletLoss().to(device)

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
        tk0 = tqdm_ext(enumerate(train_loader), total=len(train_loader), leave=True)
        optimizer.zero_grad()
        for i, (x_batch,) in tk0:
            y_pred = model(x_batch.to(device),
                           attention_mask=(x_batch > 0).to(device), labels=None)
            loss = loss_func(y_pred)
            if apex_mixed_precision:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (i + 1) % accum_steps == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()

            lossf = 0.96 * lossf + 0.04 * loss.item() if lossf else loss.item()
            tk0.set_postfix(loss=lossf, refresh=False)

            avg_loss += loss.item() / len(train_loader)
            loss_history.append(loss.item())

        tq.set_postfix(avg_loss=avg_loss)

    torch.save(model.state_dict(), output_model_file_name)

    plt.plot(range(len(loss_history)), loss_history)
    plt.ylabel('Loss')
    plt.xlabel('Batch number')
    plt.savefig('loss_plot.png', dpi=300)

    return model


if __name__ == '__main__':
    config = set_configs()

    device = torch.device(f"cuda:{config['torch_device']}")

    with timer('Read data and convert to BERT format'):
        train_loader = parse_data_to_bert_format(
            path_to_data=config['path_to_data'],
            train_file_name=config['train_file_name'],
            anchor_col_name=config['anchor_col_name'],
            pos_col_name=config['pos_col_name'],
            neg_col_name=config['neg_col_name'],
            path_to_pretrained_model=config['path_to_pretrained_model'],
            max_seq_length=config['max_seq_length'],
            batch_size=config['batch_size'],
            toy=config['toy']
        )

    with timer('Setting up BERT model'):
        model, optimizer = setup_bert_model(
            path_to_pretrained_model=config['path_to_pretrained_model'],
            epochs=config['epochs'],
            lrate=config['lrate'],
            lrate_clf=config['lrate_clf'],
            batch_size=config['batch_size'],
            accum_steps=config['accum_steps'],
            lin_dim=config['lin_dim'],
            lin_dropout_prob=config['lin_dropout_prob'],
            warmup=config['warmup'],
            apex_mixed_precision=config['apex_mixed_precision'],
            seed=config['seed'], device=device, train_loader=train_loader,
            clf_class=BertForSequencePairClassificationTriplets
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
            train_loader=train_loader
        )
