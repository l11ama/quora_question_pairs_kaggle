import sys

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, TqdmDeprecationWarning


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
