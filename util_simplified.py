from torch.optim import SGD, Adam, AdamW, RMSprop
from data import random_split, load_split_for_fold
from loss import ranking_loss, devise_loss, ranking_loss_UNCol, ranking_loss_UNRow
from compatibility import (
    dot_product_compatibility,
    euclidean_distance_compatibility,
    cosine_similarity_compatibility,
    manhattan_distance_compatibility
)
import numpy as np
import random
import torch
import audmetric

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_optimizer(optimizer_name, lr, params):
    optimizers = {
        'SGD': SGD(params=params, momentum=0.9, lr=lr),
        'Adam': Adam(params=params, lr=lr),
        'AdamW': AdamW(params=params, lr=lr, weight_decay=0.0001),
        'RMSprop': RMSprop(params=params, lr=lr, alpha=.95, eps=1e-7)
    }
    return optimizers[optimizer_name]


def get_loss_function(loss_name):
    loss_functions = {
        'ranking': ranking_loss,
        'devise': devise_loss,
        'ranking_UNCol': ranking_loss_UNCol,
        'ranking_UNRow': ranking_loss_UNRow,
    }
    return loss_functions[loss_name]


def get_compatibility_function(compatibility_name):
    compatibility_functions = {
        "dot_product": dot_product_compatibility,
        "euclidean_distance": euclidean_distance_compatibility,
        "manhattan_distance": manhattan_distance_compatibility,
        "cosine_similarity": cosine_similarity_compatibility,
    }
    return compatibility_functions[compatibility_name]


def get_splitting_function(splitting_name):
    splitting_functions = {
        "random": random_split,
        "predefined_zsl_folds": load_split_for_fold,
    }
    return splitting_functions[splitting_name]

def get_metrics():
    return {
        "ACC": audmetric.accuracy,
        "UAR": audmetric.unweighted_average_recall,
        "F1": audmetric.unweighted_average_fscore
    }