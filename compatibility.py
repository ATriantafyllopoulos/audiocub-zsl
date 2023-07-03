r"""File collecting compatibility functions.

Compatibility functions must take following arguments:

1. embeddings: the audio embeddings
2. class_embeddings: attribute embeddings for **all** candidate classes
"""

import torch


def dot_product_compatibility(
    embeddings: torch.Tensor,
    class_embeddings: torch.Tensor
):
    r"""Dot-product compatibility function.

    Suggested by H. Xia and T. Virtanen 
    (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9376628&tag=1).
    """
    return embeddings @ class_embeddings.T