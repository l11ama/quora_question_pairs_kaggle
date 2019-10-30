from pytorch_pretrained_bert import BertForSequenceClassification, BertModel
from torch import nn
import torch


class BertForSequencePairClassification(BertForSequenceClassification):

    def __init__(self, config, lin_dim, lin_dropout_prob, num_labels=1):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(2 * config.hidden_size, lin_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(lin_dim),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim, lin_dim//4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(lin_dim//4),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim//4, num_labels)
        )

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        _, pooled_output_left = self.bert(input_ids[:, :, 0],
                                          attention_mask=attention_mask[:, :, 0],
                                          token_type_ids=token_type_ids,
                                          output_all_encoded_layers=False)

        _, pooled_output_right = self.bert(input_ids[:, :, 1],
                                           attention_mask=attention_mask[:, :, 1],
                                           token_type_ids=token_type_ids,
                                           output_all_encoded_layers=False)

        pooled_outputs = torch.cat((pooled_output_left, pooled_output_right), dim=1)

        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits.squeeze(dim=1)
