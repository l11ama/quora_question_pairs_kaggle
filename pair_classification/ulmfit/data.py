import numpy as np
import pandas as pd
import torch

from fastai.basic_data import DataBunch
from fastai.core import is1d, progress_bar, BatchSamples
from fastai.text.data import *
from fastai.data_block import *
from fastai.text.data import _join_texts
from torch.utils.data import DataLoader
from fastai.torch_core import *


class PairTokenizeProcessor(TokenizeProcessor):
    """
    Tokenize Processor to tokenize
    """

    def process_one(self, item):
        return super().process_one(item[0]), super().process_one(item[1])

    def process(self, ds):
        items1 = _join_texts(ds.items[:, 0], self.mark_fields, self.include_bos, self.include_eos)
        items2 = _join_texts(ds.items[:, 1], self.mark_fields, self.include_bos, self.include_eos)
        ds.items = np.stack((items1, items2), axis=-1)

        tokens1 = []
        tokens2 = []
        for i in progress_bar(range(0, len(ds), self.chunksize), leave=False):
            tokens1 += self.tokenizer.process_all(ds.items[i:i + self.chunksize, 0])
            tokens2 += self.tokenizer.process_all(ds.items[i:i + self.chunksize, 1])
        ds.items = np.stack((tokens1, tokens2), axis=-1)


class PairNumericalizeProcessor(NumericalizeProcessor):

    def process_one(self, item):
        #         return super().process_one(item[0])
        return super().process_one(item[0]), super().process_one(item[1])


def pad_collate(samples: BatchSamples, pad_idx: int = 1, pad_first: bool = True,
                backwards: bool = False) -> Tuple[LongTensor, LongTensor]:
    "Function that collect samples and adds padding. Flips token order if needed"
    samples = to_data(samples)

    def pair_max_len_f(pair):
        return max(len(pair[0]), len(pair[1]))

    max_len = max([pair_max_len_f(s[0]) for s in samples])

    res_pair = torch.zeros(2, len(samples), max_len).long() + pad_idx
    if backwards: pad_first = not pad_first
    for i, s in enumerate(samples):
        if pad_first:
            res_pair[0, i, -len(s[0][0]):] = LongTensor(s[0][0])
            res_pair[1, i, -len(s[0][1]):] = LongTensor(s[0][1])
        else:
            res_pair[0, i, :len(s[0][0]):] = LongTensor(s[0][0])
            res_pair[1, i, :len(s[0][1]):] = LongTensor(s[0][1])
    if backwards:
        res_pair[0] = res_pair[0].flip(1)
        res_pair[1] = res_pair[1].flip(1)

    return res_pair, torch.tensor(np.array([s[1] for s in samples]))


class PairTextClasDataBunch(TextClasDataBunch):

    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path: PathOrStr = '.', bs: int = 32,
               val_bs: int = None, pad_idx=1,
               pad_first=True, device: torch.device = None, no_check: bool = False,
               backwards: bool = False, **dl_kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x,
                                       key=lambda t: max(len(datasets[0][t][0][0].data),
                                                         len(datasets[0][t][0][1].data)), bs=bs)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True,
                              **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [max(len(t[0]), len(t[1])) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, collate_fn=collate_fn, no_check=no_check)


class PairTextList(TextList):
    _bunch = PairTextClasDataBunch
    _processor = [PairTokenizeProcessor, PairNumericalizeProcessor]

    def get(self, i):
        o = self.items[i]
        return [Text(o[0], self.vocab.textify(o[0])), Text(o[1], self.vocab.textify(o[1]))]

    @classmethod
    def from_df(cls, df: pd.DataFrame, path: PathOrStr = '.', col1: int = 0, col2: int = 1,
                processor: PreProcessor = None, **kwargs) -> 'ItemList':
        "Create an `ItemList` in `path` from the inputs in the `cols` of `df`."
        inputs = df.iloc[:, [col1, col2]]
        #         inputs = df.iloc[:,df_names_to_idx([col1, col2], df)]
        assert inputs.isna().sum().sum() == 0, f"You have NaN values in column(s) {cols} of your dataframe, please fix it."
        res = cls(items=inputs.values, path=path, inner_df=df, processor=processor, **kwargs)
        return res

