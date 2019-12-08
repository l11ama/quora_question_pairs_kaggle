import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DistilBertForSequencePairBinaryClassification(nn.Module):
    def __init__(self, model_name, encoding_dim=128):
        super().__init__()

        config = AutoConfig.from_pretrained(
            model_name, num_labels=1)

        self.distilbert = AutoModel.from_pretrained(model_name,
                                                    config=config)
        self.pre_classifier = nn.Linear(config.dim, encoding_dim)
        self.classifier = nn.Linear(3 * encoding_dim, 1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def encode_sequence(self, features, mask):
        assert mask is not None, "attention mask is none"
        distilbert_output = self.distilbert(
            input_ids=features,
            attention_mask=mask
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        return pooled_output

    def forward(self, features_left, mask_left, features_right, mask_right):
        output_left = self.encode_sequence(features_left, mask_left)
        output_right = self.encode_sequence(features_right, mask_right)
        output_prod = output_left * output_right

        concat_ouputs = torch.cat((output_left, output_right, output_prod), dim=1)  # (bs, 3*dim)
        concat_ouputs = nn.ReLU()(concat_ouputs)  # (bs, 3*dim)

        concat_ouputs = self.dropout(concat_ouputs)  # (bs, 3*dim)
        logits = self.classifier(concat_ouputs)

        return logits


