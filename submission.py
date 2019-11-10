import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from preprocessing.triplets import generate_triplet_dataset
from pair_classification.bert_finetuning.util import seed_everything, timer, set_configs


def save_triplets_train(train_path, triplets_path, seed):
    train_df = pd.read_csv(train_path)
    seed_everything(seed=seed)

    X_train = np.arange(len(train_df))
    y_train = train_df['is_duplicate'].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)
    train_df = train_df.iloc[X_train]

    triplet_train_df = pd.DataFrame(generate_triplet_dataset(train_df))
    triplet_train_df.to_csv(triplets_path)


def metric_learning(config, device):
    from pair_classification.bert_finetuning.triplets.hugginface_pair_clf import BertForSequenceTriplets
    from pair_classification.bert_finetuning.triplets.finetune import parse_data_to_bert_format, train, setup_bert_model

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
            clf_class=BertForSequenceTriplets
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
            bert_freezed=False
        )

    torch.save(model.state_dict(), config['path_to_output_model'] + 'bert_triplets.bin')


def pair_classification(config, device):
    from pair_classification.bert_finetuning.finetune import parse_data_to_bert_format, \
        setup_bert_model, train, predict_for_test
    from pair_classification.bert_finetuning.hugginface_pair_clf import BertForSequencePairClassificationDSSMWithAttention

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
            toy=config['toy'],
            dssm=True
        )

    with timer('Setting up BERT model'):
        model, optimizer = setup_bert_model(
            path_to_pretrained_model=config['path_to_pretrained_model'],
            epochs=config['epochs'],
            lrate=config['lrate'], lrate_clf=config['lrate_clf'],
            batch_size=config['batch_size'],
            accum_steps=config['accum_steps'], lin_dim=config['lin_dim'],
            lin_dropout_prob=config['lin_dropout_prob'], warmup=config['warmup'],
            apex_mixed_precision=config['apex_mixed_precision'],
            seed=config['seed'], device=device, train_loader=train_loader,
            clf_class=BertForSequencePairClassificationDSSMWithAttention
        )

    encoder_state_dict = torch.load(config['path_to_output_model'] + 'bert_triplets.bin')
    model.load_state_dict(encoder_state_dict, strict=False)

    # freezed bert
    with timer('Training'):
        model = train(
            model=model,
            optimizer=optimizer,
            epochs=1,
            accum_steps=config['accum_steps'],
            apex_mixed_precision=config['apex_mixed_precision'],
            output_model_file_name=config['output_model_file_name'],
            device=device,
            train_loader=train_loader,
            batch_size=config['batch_size'],
            class_names=class_names,
            val_loader=val_loader,
            bert_freezed=True
        )

    # unfreezed bert
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
            class_names=class_names,
            val_loader=val_loader,
            bert_freezed=False
        )

    torch.save(model.state_dict(), config['path_to_output_model'] + 'bert_pair_clf_prod_with_att.bin')

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


if __name__ == '__main__':
    config = set_configs('models/bert_finetuning/config.yml')
    config_metric_learning = set_configs('models/bert_finetuning/config_triplets.yml')
    device = torch.device(f"cuda:{config['torch_device']}")

    # save csv with triplets for metric learning
    save_triplets_train(config['path_to_data'] + config['train_file_name'],
                        config_metric_learning['path_to_data'] + config_metric_learning['train_file_name'],
                        seed=config['seed'])

    metric_learning(config_metric_learning, device)

    pair_classification(config, device)
