from typing import Mapping, List
import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pair_classification.catalyst.data.nlp.pair_bin_classify import TextPairBinaryClfDataset


class TextPairBinaryClfDebiasedDataset(TextPairBinaryClfDataset):
    def __init__(self, texts_left: List[str], texts_right: List[str], weights: List[float], labels: List[str] = None,
                 max_seq_length: int = 512, model_name: str = 'distilbert-base-uncased'):
        """

        :param texts_left: a list with first in pair texts in to classify or to train the classifier on
        :param texts_right: a list with second in pair texts to classify or to train the classifier on
        :param labels: a list with classification labels (int: 0/1)
        :param max_seq_length: maximal sequence length, texts will be stripped
        :param model_name: transformer model name, we need here to perform
                           appropriate tokenization
        :param weights: sample weights for criterion

        """
        super().__init__(texts_left, texts_right, labels, max_seq_length, model_name)
        self.weights = weights

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:

        output_dict = super().__getitem__(index)

        # sample weights
        if self.weights is not None:
            weight = self.weights[index]
            weight_encoded = torch.Tensor([weight]).float()
            output_dict['weights'] = weight_encoded

        return output_dict
