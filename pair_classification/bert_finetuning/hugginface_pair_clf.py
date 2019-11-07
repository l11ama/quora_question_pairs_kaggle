from pytorch_pretrained_bert import BertForSequenceClassification, BertModel
from torch import nn
import torch

from pair_classification.bert_finetuning.util import mask_softmax


class BertForSequencePairClassification(BertForSequenceClassification):

    def __init__(self, config, lin_dim, lin_dropout_prob, num_labels=1):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, lin_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(lin_dim),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim, lin_dim//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(lin_dim//2),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim//2, num_labels)
        )

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output).squeeze(dim=1)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.view(-1))
            return loss
        else:
            return logits


class BertForSequencePairClassificationConcat(BertForSequenceClassification):

    def __init__(self, config, lin_dim, lin_dropout_prob, num_labels=1):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, lin_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(lin_dim),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim, lin_dim//2),
            nn.ReLU(inplace=True))

        self.pair_classifier = nn.Sequential(
            nn.BatchNorm1d(lin_dim),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim, num_labels)
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

        output_left = self.dropout(pooled_output_left)
        output_right = self.dropout(pooled_output_right)

        output_left = self.classifier(output_left)
        output_right = self.classifier(output_right)

        outputs = torch.cat((output_left, output_right), dim=1)

        logits = self.pair_classifier(outputs)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits.squeeze(dim=1)


class BertForSequencePairClassificationDSSM(BertForSequenceClassification):

    def __init__(self, config, lin_dim, lin_dropout_prob, num_labels=1):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, lin_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(lin_dim),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim, lin_dim//2),
            nn.ReLU(inplace=True))

        self.pair_classifier = nn.Sequential(
            nn.BatchNorm1d(lin_dim//2),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim//2, num_labels)
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

        output_left = self.dropout(pooled_output_left)
        output_right = self.dropout(pooled_output_right)

        output_left = self.classifier(output_left)
        output_right = self.classifier(output_right)

        outputs = output_left * output_right

        logits = self.pair_classifier(outputs)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits.squeeze(dim=1)


class BertForSequencePairClassificationDSSMWithAttention(BertForSequenceClassification):

    def __init__(self, config, lin_dim, lin_dropout_prob, num_labels=1):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier_fc_attention = nn.Linear(config.hidden_size, 1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(2*config.hidden_size, lin_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(lin_dim),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim, lin_dim // 2),
            nn.ReLU(inplace=True))

        self.pair_classifier = nn.Sequential(
            nn.BatchNorm1d(lin_dim // 2),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim // 2, num_labels)
        )

        self.apply(self.init_bert_weights)

    def attend_to_other_output(self, sequence_output, mask, pooled_output_other):
        att = self.classifier_fc_attention(torch.tanh(sequence_output + pooled_output_other)).squeeze(-1)  # [b,msl,h*2]->[b,msl]
        att = mask_softmax(att, mask)  # [b,msl]
        return torch.sum(att.unsqueeze(-1) * sequence_output, dim=1)  # [b,h*2]

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        sequence_output_left, pooled_output_left = self.bert(input_ids[:, :, 0],
                                                             attention_mask=attention_mask[:, :, 0],
                                                             token_type_ids=token_type_ids,
                                                             output_all_encoded_layers=False)

        sequence_output_right, pooled_output_right = self.bert(input_ids[:, :, 1],
                                                               attention_mask=attention_mask[:, :, 1],
                                                               token_type_ids=token_type_ids,
                                                               output_all_encoded_layers=False)

        attended_left = self.attend_to_other_output(sequence_output_left, attention_mask[:, :, 0], pooled_output_right)
        attended_right = self.attend_to_other_output(sequence_output_right, attention_mask[:, :, 1], pooled_output_left)

        output_left = self.dropout(torch.cat([attended_left, pooled_output_left], dim=1))
        output_right = self.dropout(torch.cat([attended_right, pooled_output_right], dim=1))

        output_left = self.classifier(output_left)
        output_right = self.classifier(output_right)

        outputs = output_left * output_right

        logits = self.pair_classifier(outputs)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits.squeeze(dim=1)
