import torch

from torch.nn import BCEWithLogitsLoss


class SampleWeightedBCELoss(BCEWithLogitsLoss):
    def __init__(
        self,
        *args,
        target_name: str = "targets",
        weight_name: str = "weights",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.target_name = target_name
        self.weight_name = weight_name
        self.reduction = "none"

    def forward(self, input, target_with_weight):
        target = target_with_weight[self.target_name]
        weight = target_with_weight[self.weight_name]

        loss = super().forward(input, target)
        loss = loss * weight
        loss = torch.mean(loss)
        return loss
