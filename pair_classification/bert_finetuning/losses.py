import torch


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1):
        """
        Basic Triplet Loss as proposed in 'FaceNet: A Unified Embedding for Face Recognition and Clustering'
        Args:
            margin:             float, Triplet Margin - Ensures that positives aren't placed arbitrarily close to the anchor.
                                Similarly, negatives should not be placed arbitrarily far away.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def triplet_distance(self, anchor, positive, negative):
        """
        Compute triplet loss.
        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.nn.functional.relu((anchor-positive).pow(2).sum(dim=1) - (anchor-negative).pow(2).sum(dim=1) + self.margin)

    def forward(self, triplet):
        """
        Args:
            triplet embeddings
        Returns:
            triplet loss (torch.Tensor(), batch-averaged)
        """
        #Compute triplet loss
        anchor, positive, negative = triplet
        loss = self.triplet_distance(anchor, positive, negative)

        return torch.mean(loss)
