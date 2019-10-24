from fastai.text import *
from fastai.text.learner import _model_meta
from fastai.torch_core import *


model_meta = _model_meta


class PairMultiBatchEncoder(MultiBatchEncoder):

    def forward(self, input: LongTensor):
        """
        :param input: of shape `(batch_size, seq_len)`
        :return: tuple((raw_outputs1,outputs1,mask1),(raw_outputs2,outputs2,mask2))
                 layer rnn outputs for pair of samples
        """
        res1 = super().forward(input[0])
        res2 = super().forward(input[1])
        
        return res1, res2, input[2].float()


def pool(input: Tuple[Tensor, Tensor, Tensor]):
    raw_outputs, outputs, mask = input
    output = outputs[-1]
    avg_pool = output.masked_fill(mask[:, :, None], 0).mean(dim=1)
    avg_pool *= output.size(1) / (output.size(1) - mask.type(avg_pool.dtype).sum(dim=1))[:, None]
    max_pool = output.masked_fill(mask[:, :, None], -float('inf')).max(dim=1)[0]
    x = torch.cat([output[:, -1], max_pool, avg_pool], 1)
    return x, raw_outputs, outputs


def concat(t1: List[Tensor], t2: List[Tensor]) -> List[Tensor]:
    "Concatenate rnn outputs for pair along batch dim"
    return [torch.cat([t1[layer], t2[layer]], dim=0) for layer in range(len(t1))]


class PairPoolingDSSMClassifier(PoolingLinearClassifier):

    def __init__(self, layers:Collection[int], drops:Collection[float]):
        super(PoolingLinearClassifier, self).__init__()
        mod_layers = []
        if len(drops) != len(layers)-1:
            raise ValueError("Number of layers and dropout values do not match.")
        layers = layers[:-1]
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        drops = drops[:-1]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def forward(self, input) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """input:Tuple[Tuple[Tensor,Tensor, Tensor],
                       Tuple[Tensor,Tensor, Tensor]] - encodings for pair of documents
        """

        enc1, raw_out1, out1 = pool(input[0])
        enc2, raw_out2, out2 = pool(input[1])

        enc1 = self.layers(enc1)
        enc2 = self.layers(enc2)
        
        x = torch.sum(enc1 * enc2, dim=-1)
        x = torch.stack([-x, x], dim=-1)
        
        return x, concat(raw_out1, raw_out2), concat(out1, out2)

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
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result

class PairAttentionPoolingDSSMClassifier(PoolingLinearClassifier):

    def __init__(self, layers:Collection[int], drops:Collection[float]):
        super(PoolingLinearClassifier, self).__init__()
        mod_layers = []
        if len(drops) != len(layers)-1:
            raise ValueError("Number of layers and dropout values do not match.")
        layers = layers[:-1]
        activations = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        drops = drops[:-1]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activations):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)
        self.fc_attention = nn.Linear(layers[0]//3, 1) #get top layer hidden size from lstm

    def pool(self, input: Tuple[Tensor, Tensor, Tensor]):
        raw_outputs, outputs, mask = input
        output = outputs[-1]
        avg_pool = output.masked_fill(mask[:, :, None], 0).mean(dim=1)
        avg_pool *= output.size(1) / (output.size(1) - mask.type(avg_pool.dtype).sum(dim=1))[:, None]
        max_pool = output.masked_fill(mask[:, :, None], -float('inf')).max(dim=1)[0]


        att = self.fc_attention(output).squeeze(-1)  # [b,msl,h*2]->[b,msl]
        att = mask_softmax(att, mask)  # [b,msl]
        r_att = torch.sum(att.unsqueeze(-1) * output, dim=1)  # [b,h*2]

        x = torch.cat([r_att, max_pool, avg_pool], 1)
        return x, raw_outputs, outputs        
        
    def forward(self, input) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """input:Tuple[Tuple[Tensor,Tensor, Tensor],
                       Tuple[Tensor,Tensor, Tensor]] - encodings for pair of documents
        """

        enc1, raw_out1, out1 = self.pool(input[0])
        enc2, raw_out2, out2 = self.pool(input[1])

        enc1 = self.layers(enc1)
        enc2 = self.layers(enc2)
        
#         enc1_normed = enc1 / torch.sqrt(torch.sum(enc1 * enc1, dim=1, keepdim=True))
#         enc2_normed = enc2 / torch.sqrt(torch.sum(enc2 * enc2, dim=1, keepdim=True))
        x = torch.sum(enc1 * enc2, dim=-1)
        x = torch.stack([torch.zeros_like(x), x], dim=-1)
        
        return x, concat(raw_out1, raw_out2), concat(out1, out2)

    
class PairAttentionPoolingDSSMFC1Classifier(PoolingLinearClassifier):

    def __init__(self, layers:Collection[int], drops:Collection[float]):
        super(PoolingLinearClassifier, self).__init__()
        mod_layers = []
        if len(drops) != len(layers)-1:
            raise ValueError("Number of layers and dropout values do not match.")
        layers = layers[:-1]
        activations = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        drops = drops[:-1]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activations):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)
        attention_size = layers[0]//3 #get top layer hidden size from lstm
        self.fc_attention = nn.Linear(attention_size, 1)
        
        self.pair_linear = nn.Linear(layers[-1], 1)

    def pool(self, input: Tuple[Tensor, Tensor, Tensor]):
        raw_outputs, outputs, mask = input
        output = outputs[-1]
        avg_pool = output.masked_fill(mask[:, :, None], 0).mean(dim=1)
        avg_pool *= output.size(1) / (output.size(1) - mask.type(avg_pool.dtype).sum(dim=1))[:, None]
        max_pool = output.masked_fill(mask[:, :, None], -float('inf')).max(dim=1)[0]


        att = self.fc_attention(output).squeeze(-1)  # [b,msl,h*2]->[b,msl]
        att = mask_softmax(att, mask)  # [b,msl]
        r_att = torch.sum(att.unsqueeze(-1) * output, dim=1)  # [b,h*2]

        x = torch.cat([r_att, max_pool, avg_pool], 1)
        return x, raw_outputs, outputs        
        
    def forward(self, input) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """input:Tuple[Tuple[Tensor,Tensor, Tensor],
                       Tuple[Tensor,Tensor, Tensor]] - encodings for pair of documents
        """

        enc1, raw_out1, out1 = self.pool(input[0])
        enc2, raw_out2, out2 = self.pool(input[1])

        enc1 = self.layers(enc1)
        enc2 = self.layers(enc2)
                
        enc = enc1 * enc2
        x = self.pair_linear(enc)
        x = torch.stack([torch.zeros_like(x), x], dim=-1)
        
        return x, concat(raw_out1, raw_out2), concat(out1, out2)


class PairPoolingLinearClassifier(PoolingLinearClassifier):

    def forward(self, input) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """input:Tuple[Tuple[Tensor,Tensor, Tensor],
                       Tuple[Tensor,Tensor, Tensor]] - encodings for pair of documents
        """

        enc1, raw_out1, out1 = pool(input[0])
        enc2, raw_out2, out2 = pool(input[1])

        x = self.layers(enc1 * enc2)
        
        return x, concat(raw_out1, raw_out2), concat(out1, out2)


def get_text_classifier(arch: Callable, vocab_sz: int, n_class: int, bptt: int = 70,
                        max_len: int = 20 * 70, config: dict = None,
                        drop_mult: float = 1., lin_ftrs: Collection[int] = None,
                        ps: Collection[float] = None,
                        pad_idx: int = 1,
                        encoder_class=PairMultiBatchEncoder,
                        clf_class=PairPoolingLinearClassifier) -> nn.Module:
    "Create a text classifier from `arch` and its `config`, maybe `pretrained`."
    meta = model_meta[arch]
    config = ifnone(config, meta['config_clas'].copy())
    for k in config.keys():
        if k.endswith('_p'): config[k] *= drop_mult
    if lin_ftrs is None: lin_ftrs = [50]
    if ps is None:  ps = [0.1]
    layers = [config[meta['hid_name']] * 3] + lin_ftrs + [n_class]
    ps = [config.pop('output_p')] + ps
    init = config.pop('init') if 'init' in config else None
    encoder = encoder_class(bptt, max_len, arch(vocab_sz, **config), pad_idx=pad_idx)
    model = SequentialRNN(encoder, clf_class(layers, ps))
    return model if init is None else model.apply(init)


def text_classifier_learner(data: DataBunch, arch: Callable, bptt: int = 70, max_len: int = 70 * 20,
                            config: dict = None,
                            pretrained: bool = True, drop_mult: float = 1.,
                            lin_ftrs: Collection[int] = None,
                            ps: Collection[float] = None,
                            encoder_class=PairMultiBatchEncoder,
                            clf_class=PairPoolingLinearClassifier,
                            **learn_kwargs) -> 'TextClassifierLearner':
    "Create a `Learner` with a text classifier from `data` and `arch`."
    model = get_text_classifier(arch, len(data.vocab.itos), data.c, bptt=bptt, max_len=max_len,
                                config=config, drop_mult=drop_mult, lin_ftrs=lin_ftrs, ps=ps,
                                encoder_class=encoder_class, clf_class=clf_class)
    meta = model_meta[arch]
    learn = RNNLearner(data, model, split_func=meta['split_clas'], **learn_kwargs)
    if pretrained:
        if 'url' not in meta:
            warn("There are no pretrained weights for that architecture yet!")
            return learn
        model_path = untar_data(meta['url'], data=False)
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn.load_pretrained(*fnames, strict=False)
        learn.freeze()
    return learn
