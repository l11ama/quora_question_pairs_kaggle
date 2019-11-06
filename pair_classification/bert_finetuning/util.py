import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, TqdmDeprecationWarning
from contextlib import contextmanager
import random
import time
import os

import torch

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


def sigmoid_np(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

NEG_INF = -10000
TINY_FLOAT = 1e-6


def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((~mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result


class tqdm_ext(tqdm):

    N_PRINTS = 100

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        print_every = self.total // self.N_PRINTS + 1
        n = self.n
        for obj in iterable:
            yield obj
            # Update and possibly print the progressbar.
            n += 1
            if n % print_every == 0:
                print('Iter {}/{}. Postfix = {}'.format(n, self.total, self.postfix))

        # Closing the progress bar.
        # Update some internal variables for close().
        self.n = n
        self.close()
