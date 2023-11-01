import os
import random

import audmetric
import numpy as np
import pandas as pd
import torch
from torch.optim import SGD, Adam, AdamW, RMSprop

from compatibility import (
    dot_product_compatibility,
    euclidean_distance_compatibility,
    cosine_similarity_compatibility,
    manhattan_distance_compatibility
)
from data import random_split, load_split_for_fold
from loss import ranking_loss, devise_loss, ranking_loss_UNCol, ranking_loss_UNRow


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
        "predefined": load_split_for_fold,
    }
    return splitting_functions[splitting_name]


def predefined(split_dir, fold_id, label):
    target_dir = os.path.join(split_dir, str(fold_id))

    train = pd.read_csv(os.path.join(target_dir, 'train.csv'))[label].tolist()
    dev = pd.read_csv(os.path.join(target_dir, 'dev.csv'))[label].tolist()
    test = pd.read_csv(os.path.join(target_dir, 'test.csv'))[label].tolist()

    return train, dev, test


def get_metrics():
    return {
        "ACC": audmetric.accuracy,
        "UAR": audmetric.unweighted_average_recall,
        "F1": audmetric.unweighted_average_fscore
    }


def get_meta_source(meta_type, cfg):
    meta_sources = {
        'image': cfg.meta.image_features,
        'numeric': cfg.meta.numeric_features,
        'text': cfg.meta.text_features,
    }
    return meta_sources[meta_type]
