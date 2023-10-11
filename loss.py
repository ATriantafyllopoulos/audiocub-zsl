r"""File collecting loss functions.

Loss functions must take following arguments:

1. embeddings: the audio embeddings
2. class_embeddings: attribute embeddings for **all** candidate classes
3. targets: tensor with ints denoting correct class
4. compatibility_function: collable which accepts embeddings and class_embeddings
"""

import torch
import typing


def devise_loss(
    embeddings: torch.Tensor,
    class_embeddings: torch.Tensor,
    targets: torch.Tensor,
    compatibility_function: typing.Callable,
    model: torch.nn.Module
):
    r"""Ranking-based loss function.

    Suggested by A. Frome, G. Corrado, et al. in their paper "DeViSE: A Deep Visual-Semantic Embedding Model"
    (https://proceedings.neurips.cc/paper/2013/file/7cce53cf90577442771720a370c3c723-Paper.pdf).
    """
    # unit normalise the class embeddings
    norms = torch.linalg.norm(class_embeddings, dim=(-1))
    class_embeddings = class_embeddings / norms.unsqueeze(1)

    # calculate loss for each sample in the batch
    margin = 1
    loss_vector = torch.Tensor((embeddings.size(0)))
    for i in range(embeddings.size(0)):
        label = targets[i]
        emb = embeddings[i, :]
        t_label = class_embeddings[label, :]
        term1 = torch.dot(t_label, emb)

        row_sum = 0
        for t in targets:
            if t == label:
                continue
            t_j = class_embeddings[t, :]
            term2 = torch.dot(t_j, emb)
            row_sum += torch.maximum(torch.zeros((1), device=embeddings.get_device()), margin - term1 + term2)            
        loss_vector[i] = row_sum
    return loss_vector
        


def ranking_loss(
    embeddings: torch.Tensor,
    class_embeddings: torch.Tensor,
    targets: torch.Tensor,
    compatibility_function: typing.Callable,
    model: torch.nn.Module
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

    ### Try out unit norm of the text embeddings
    compatibility = compatibility_function(embeddings, class_embeddings)
    # compute ranks of compatibility matrix (double argsort)
    # see: https://stackoverflow.com/a/6266510
    # then turn them to 1-indexed by adding 1
    # which allows to compute the penalty without errors
    ranks = compatibility.argsort(axis=1).argsort(axis=1) + 1 # gets the ranks yes, but is  +1 ok ? Yes, torch.arange(1, 1) is empty => .sum() = 0
    
    # take ranks of correct class (https://stackoverflow.com/a/67951672)
    class_ranks = ranks[torch.arange(ranks.size(0)), targets]
    
    # compute 1 / rank.sum() as penalty for each class
    penalties = torch.Tensor([(1 / torch.arange(1, x)).sum() for x in class_ranks]) # consider ß(0) = 0 ### here: (1 / []).sum() = 0
    penalties = penalties.to(embeddings.get_device())
    
    # compute multiplying factor for each element
    # defined as the penalties divided by the class ranks
    # factors = penalties / class_ranks
    class_ranks -= 1
    factors = torch.tensor([penalty / c_rank if c_rank > 0 else 0 for penalty, c_rank in zip(penalties, class_ranks)], device=embeddings.get_device(), dtype=torch.float32)
    # set elements where class_rank was 0 to 0 
    # i.e. those elements that were identifed correctly => here that would be the one with rank = 1, since we added +1 above
    factors[class_ranks == 0] = 0   # but it is never == 0, since we do +1 for the ranks above and the class_ranks are inferred from the ranks, it's only 0, since some pnealties are 0 => 0 / class_rank = 0
    
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
    ### Calculate the frobenius norm
    # W = model.weight
    # frobenius = 0.1 * torch.linalg.norm(W, ord='fro')**2
    total_loss = factors * torch.maximum(hinge_loss, torch.Tensor([0]).to(embeddings.get_device())).sum(dim=1) # + frobenius # the frobenius norm is missing
    return total_loss



def ranking_loss_UNRow(
    embeddings: torch.Tensor,
    class_embeddings: torch.Tensor,
    targets: torch.Tensor,
    compatibility_function: typing.Callable,
    model: torch.nn.Module
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

    ### Try out unit norm of the class embeddings per row
    norms = torch.linalg.norm(class_embeddings, dim=(-1))
    class_embeddings = class_embeddings / norms.unsqueeze(1)
    compatibility = compatibility_function(embeddings, class_embeddings)
    # compute ranks of compatibility matrix (double argsort)
    # see: https://stackoverflow.com/a/6266510
    # then turn them to 1-indexed by adding 1
    # which allows to compute the penalty without errors
    ranks = compatibility.argsort(axis=1).argsort(axis=1) + 1 # gets the ranks yes, but is  +1 ok ? Yes, torch.arange(1, 1) is empty => .sum() = 0
    
    # take ranks of correct class (https://stackoverflow.com/a/67951672)
    class_ranks = ranks[torch.arange(ranks.size(0)), targets]
    
    # compute 1 / rank.sum() as penalty for each class
    penalties = torch.Tensor([(1 / torch.arange(1, x)).sum() for x in class_ranks]) # consider ß(0) = 0 ### here: (1 / []).sum() = 0
    penalties = penalties.to(embeddings.get_device())
    
    # compute multiplying factor for each element
    # defined as the penalties divided by the class ranks
    # factors = penalties / class_ranks
    class_ranks -= 1
    factors = torch.tensor([penalty / c_rank if c_rank > 0 else 0 for penalty, c_rank in zip(penalties, class_ranks)], device=embeddings.get_device(), dtype=torch.float32)
    # set elements where class_rank was 0 to 0 
    # i.e. those elements that were identifed correctly => here that would be the one with rank = 1, since we added +1 above
    factors[class_ranks == 0] = 0   # but it is never == 0, since we do +1 for the ranks above and the class_ranks are inferred from the ranks, it's only 0, since some pnealties are 0 => 0 / class_rank = 0
    
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
    ### Calculate the frobenius norm
    # W = model.weight
    # frobenius = 0.1 * torch.linalg.norm(W, ord='fro')**2
    total_loss = factors * torch.maximum(hinge_loss, torch.Tensor([0]).to(embeddings.get_device())).sum(dim=1) # + frobenius # the frobenius norm is missing
    return total_loss



def ranking_loss_UNCol(
    embeddings: torch.Tensor,
    class_embeddings: torch.Tensor,
    targets: torch.Tensor,
    compatibility_function: typing.Callable,
    model: torch.nn.Module
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

    ### Try out unit norm of the text embeddings per column
    norms = torch.linalg.norm(class_embeddings, dim=(-2), keepdim=True)
    class_embeddings = class_embeddings / norms.unsqueeze(1)
    compatibility = compatibility_function(embeddings, class_embeddings)
    # compute ranks of compatibility matrix (double argsort)
    # see: https://stackoverflow.com/a/6266510
    # then turn them to 1-indexed by adding 1
    # which allows to compute the penalty without errors
    ranks = compatibility.argsort(axis=1).argsort(axis=1) + 1 # gets the ranks yes, but is  +1 ok ? Yes, torch.arange(1, 1) is empty => .sum() = 0
    

    # take ranks of correct class (https://stackoverflow.com/a/67951672)
    class_ranks = ranks[torch.arange(ranks.size(0)), targets]
    
    # compute 1 / rank.sum() as penalty for each class
    penalties = torch.Tensor([(1 / torch.arange(1, x)).sum() for x in class_ranks]) # consider ß(0) = 0 ### here: (1 / []).sum() = 0
    penalties = penalties.to(embeddings.get_device())
    
    # compute multiplying factor for each element
    # defined as the penalties divided by the class ranks
    # factors = penalties / class_ranks
    class_ranks -= 1
    factors = torch.tensor([penalty / c_rank if c_rank > 0 else 0 for penalty, c_rank in zip(penalties, class_ranks)], device=embeddings.get_device(), dtype=torch.float32)
    
    # set elements where class_rank was 0 to 0 
    # i.e. those elements that were identifed correctly => here that would be the one with rank = 1, since we added +1 above
    factors[class_ranks == 0] = 0   # but it is never == 0, since we do +1 for the ranks above and the class_ranks are inferred from the ranks, it's only 0, since some pnealties are 0 => 0 / class_rank = 0
    
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
    ### Calculate the frobenius norm
    # W = model.weight
    # frobenius = 0.1 * torch.linalg.norm(W, ord='fro')**2
    total_loss = factors * torch.maximum(hinge_loss, torch.Tensor([0]).to(embeddings.get_device())).sum(dim=1) # + frobenius # the frobenius norm is missing
    return total_loss



def ranking_loss_og(
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