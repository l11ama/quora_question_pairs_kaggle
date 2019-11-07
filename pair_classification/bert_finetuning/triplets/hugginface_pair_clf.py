from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch import nn


class BertForSequenceTriplets(BertPreTrainedModel):

    def __init__(self, config,  lin_dim, lin_dropout_prob, num_labels=1):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        _, pooled_output_0 = self.bert(input_ids[:, :, 0],
                                       attention_mask=attention_mask[:, :, 0],
                                       token_type_ids=token_type_ids,
                                       output_all_encoded_layers=False)

        _, pooled_output_1 = self.bert(input_ids[:, :, 1],
                                       attention_mask=attention_mask[:, :, 1],
                                       token_type_ids=token_type_ids,
                                       output_all_encoded_layers=False)

        _, pooled_output_2 = self.bert(input_ids[:, :, 2],
                                       attention_mask=attention_mask[:, :, 2],
                                       token_type_ids=token_type_ids,
                                       output_all_encoded_layers=False)

        anchor = self.dropout(pooled_output_0)
        pos = self.dropout(pooled_output_1)
        neg = self.dropout(pooled_output_2)

        return anchor, pos, neg


class BertForSequenceTripletsWithLinear(BertForSequenceTriplets):

    def __init__(self, config,  lin_dim, lin_dropout_prob, num_labels=1):
        super().__init__(config, lin_dim, lin_dropout_prob, num_labels)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, lin_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(lin_dim),
            nn.Dropout(p=lin_dropout_prob),
            nn.Linear(lin_dim, lin_dim//2),
            nn.ReLU(inplace=True))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        anchor, pos, neg = super().forward(input_ids, token_type_ids, attention_mask, labels)

        anchor = self.classifier(anchor)
        pos = self.classifier(pos)
        neg = self.classifier(neg)

        return anchor, pos, neg
