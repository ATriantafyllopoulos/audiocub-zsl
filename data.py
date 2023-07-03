import audobject
import numpy as np
import pandas as pd
import random
import torch
import typing


class Dataset(torch.utils.data.Dataset):
    r"""Torch dataset for ZSL.

    Accepts as input one DataFrame.
    Returns audio and target class.
    Optionally transforms features 
    using `transform` arguments.

    Warning: dataframe should only have
    a `species` column as metadata.
    Every other column will be considered a feature.
    """
    def __init__(
        self,
        audio: pd.DataFrame,
        transform: typing.Callable
    ):
        self.audio = audio
        self._audio_names = list(
            set(self.audio.columns) - set(["species"])
        )
        self.transform = transform

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, item):
        species = self.audio.loc[item, "species"]
        audio = self.audio.loc[item, self._audio_names].values
        if self.transform is not None:
            audio = self.transform(audio)
        return audio.astype(np.float32), species


class LabelEncoder(audobject.Object):
    r"""Helper class to map labels."""
    def __init__(self, labels, codes=None):
        self.labels = sorted(labels)
        if codes is None:
            codes = list(range(len(labels)))
        self.codes = codes
        self.inverse_map = {code: label for code,
                    label in zip(codes, labels)}
        self.map = {label: code for code,
                            label in zip(codes, labels)}

    def encode(self, x):
        return self.map[x]

    def decode(self, x):
        return self.inverse_map[x]


class Standardizer(audobject.Object):
    r"""Helper class to normalize features."""
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.tolist()
        self.std = std.tolist()
        self._mean = mean
        self._std = std
    
    def encode(self, x):
        return (x - self._mean) / (self._std)

    def decode(self, x):
        return x * self._std + self._mean

    def __call__(self, x):
        return self.encode(x)


def random_split(species, test_percentage=0.1):
    r"""Utility function used to split data.

    Accepts as input a list of species and a percentage
    and creates three disjoint lists with species
    to train, validate, and test on.
    """
    test_species = random.sample(species, int(len(species) * test_percentage))
    other_species = list(set(species) - set(test_species))

    dev_species = random.sample(other_species, int(len(other_species) * test_percentage))
    train_species = list(set(other_species) - set(dev_species))
    return train_species, dev_species, test_species