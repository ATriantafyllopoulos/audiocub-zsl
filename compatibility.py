r"""File collecting compatibility functions.

Compatibility functions must take following arguments:

1. embeddings: the audio embeddings
2. class_embeddings: attribute embeddings for **all** candidate classes
"""

import torch
import torch.nn.functional as F

def dot_product_compatibility(
    embeddings: torch.Tensor,
    class_embeddings: torch.Tensor
):
    r"""Dot-product compatibility function.

    Suggested by H. Xia and T. Virtanen 
    (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9376628&tag=1).
    """
    return embeddings @ class_embeddings.T # Dimensions: (batchsize x embeddings) @ (embeddings, classes) = (batchsize x classes)


def euclidean_distance_compatibility(
  embeddings: torch.Tensor,
  class_embeddings: torch.Tensor  
):
    r"""Euclidean distance compatibility function.
    """
    out = torch.cdist(embeddings, class_embeddings, p=2, compute_mode='use_mm_for_euclid_dist')
    return out


def manhattan_distance_compatibility(
  embeddings: torch.Tensor,
  class_embeddings: torch.Tensor  
):
    r"""Manhattan distance compatibility function.
    """
    out = torch.cdist(embeddings, class_embeddings, p=1, compute_mode='use_mm_for_euclid_dist')
    return out


def cosine_similarity_compatibility(
    embeddings: torch.Tensor,
    class_embeddings: torch.Tensor 
):
    r"""Cosine similarity compatibility function.
    """
    batch_size = embeddings.size(0)
    num_classes = class_embeddings.size(0)
    cosine_similarity = torch.zeros(size=(batch_size, num_classes), device=embeddings.get_device())
    for b in range(batch_size):
        for c in range(num_classes):
            cosine_similarity[b, c] = F.cosine_similarity(embeddings[b, :], class_embeddings[c, :], dim=0)

    return cosine_similarity



if __name__=='__main__':
    batch_size = 256
    embedding_length = 768
    num_classes = 78
    emb = torch.randn(size=(batch_size, embedding_length))
    class_emb = torch.randn(size=(num_classes, embedding_length))

    cosine_similarity = torch.zeros((batch_size, num_classes))
    for b in range(batch_size):
        for c in range(num_classes):
            cosine_similarity[b, c] = F.cosine_similarity(emb[b, :], class_emb[c, :], dim=0)
    
    print(cosine_similarity)
    print(cosine_similarity.shape)
    