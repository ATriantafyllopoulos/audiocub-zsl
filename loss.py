r"""File collecting loss functions.

Loss functions must take following arguments:

1. embeddings: the audio embeddings
2. class_embeddings: attribute embeddings for **all** candidate classes
3. targets: tensor with ints denoting correct class
4. compatibility_function: collable which accepts embeddings and class_embeddings
"""

import torch
import typing


def ranking_loss(
    embeddings: torch.Tensor,
    class_embeddings: torch.Tensor,
    targets: torch.Tensor,
    compatibility_function: typing.Callable
):
    r"""Ranking-based loss function.

    Suggested by H. Xia and T. Virtanen 
    (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9376628&tag=1).

    Process follows these steps:

    1. First compute the compatibility between audio and class embeddings
    2. Then compute the predicted ranks for each element in the batch
    3. Compute the penalties for each predicted element
      - Penalties are computed as 1/a for a in range(1, rank(y))
    4. Compute Hinge loss using compatibility function
      - Computed as \sum_{over all y} {\delta{y, y_n} + comp|y - comp|y_n}
        + Where y_n is the target class for current instance

    """
    # compute compatibility between output and class embeddings
    compatibility = compatibility_function(embeddings, class_embeddings)
    # compute ranks of compatibility matrix (double argsort)
    # see: https://stackoverflow.com/a/6266510
    # then turn them to 1-indexed by adding 1
    # which allows to compute the penalty without errors
    ranks = compatibility.argsort(axis=1).argsort(axis=1) + 1

    # take ranks of correct class (https://stackoverflow.com/a/67951672)
    class_ranks = ranks[torch.arange(ranks.size(0)), targets]

    # compute 1 / rank.sum() as penalty for each class
    penalties = torch.Tensor([(1 / torch.arange(1, x)).sum() for x in class_ranks])
    penalties = penalties.to(embeddings.get_device())

    # compute multiplying factor for each element
    # defined as the penalties divided by the class ranks
    factors = penalties / class_ranks
    # set elements where class_rank was 0 to 0 
    # i.e. those elements that were identifed correctly
    factors[class_ranks == 0] = 0

    ######################################
    # Compute Hinge loss
    ######################################
    # set up delta function (1 whenever y==y_hat)
    deltas = torch.ones(compatibility.shape).to(embeddings.get_device())
    deltas[torch.arange(deltas.size(0)), targets] = 0
    # take compatibility values of correct class
    # expand them to have a matrix of 1 value for each candidate class
    class_compatibilities = compatibility[torch.arange(compatibility.size(0)), targets].repeat(compatibility.shape[1], 1).T
    # compute hinge loss
    hinge_loss = deltas + compatibility - class_compatibilities
    # weigh hinge loss for each element by the normalized rank penalty
    total_loss = factors * torch.maximum(hinge_loss, torch.Tensor([0]).to(embeddings.get_device())).sum(dim=1)
    return total_loss