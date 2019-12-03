from typing import Mapping, List
import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextPairBinaryClfDataset(Dataset):
    def __init__(self,
                 texts_left: List[str],
                 texts_right: List[str],
                 labels: List[str] = None,
                 max_seq_length: int = 512,
                 model_name: str = 'distilbert-base-uncased'):
        """

        :param texts_left: a list with first in pair texts in to classify or to train the classifier on
        :param texts_right: a list with second in pair texts to classify or to train the classifier on
        :param labels: a list with classification labels (int: 0/1)
        :param max_seq_length: maximal sequence length, texts will be stripped
        :param model_name: transformer model name, we need here to perform
                           appropriate tokenization

        """

        self.texts_left = texts_left
        self.texts_right = texts_right
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # suppresses tokenizer warnings
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.FATAL)

        # special tokens for transformers
        # in the simplest case a [CLS] token is added in the beginning
        # and [SEP] token is added in the end of a piece of text
        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]

    def __len__(self):
        return len(self.texts_left)

    def encode_text(self, text):
        text_encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).squeeze(0)

        # padding short texts
        true_seq_length = text_encoded.size(0)
        pad_size = self.max_seq_length - true_seq_length
        pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
        text_tensor = torch.cat((text_encoded, pad_ids))

        # dealing with attention masks - there's a 1 for each input token and
        # if the sequence is shorter that `max_seq_length` then the rest is
        # padded with zeroes. Attention mask will be passed to the model in
        # order to compute attention scores only with input data ignoring padding
        mask = torch.ones_like(text_encoded, dtype=torch.int8)
        mask_pad = torch.zeros_like(pad_ids, dtype=torch.int8)
        mask = torch.cat((mask, mask_pad))

        return text_tensor, mask

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:

        x_left, mask_left = self.encode_text(self.texts_left[index])
        x_right, mask_right = self.encode_text(self.texts_right[index])

        output_dict = {
            'features_left': x_left,
            'mask_left': mask_left,
            'features_right': x_right,
            'mask_right': mask_right
        }

        # encoding target
        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([y]).float()
            output_dict["targets"] = y_encoded

        return output_dict
