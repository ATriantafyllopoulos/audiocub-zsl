import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import Dataset, LabelEncoder, Standardizer
from util_simplified import (
    get_optimizer,
    get_compatibility_function,
    get_loss_function,
    get_splitting_function,
    get_meta_source,
    set_random_seed, get_metrics
)


def train_epoch(loader, model, class_emb, optimizer, device, comp_func, loss_func, writer, epoch):
    model.train()
    model.to(device)

    for index, (audio, targets) in tqdm(enumerate(loader), total=len(loader), desc="Training", disable=True):
        embeddings = model(audio.to(device))

        loss = loss_func(embeddings=embeddings, class_embeddings=class_emb, targets=targets,
                         compatibility_function=comp_func, model=model).mean()

        if index % 50 == 0:
            writer.add_scalar('Loss', loss, epoch * len(loader) + index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(loader, model, class_embeddings, device, comp_func, metrics):
    model.eval()
    model.to(device)

    predictions, targets = [], []
    for index, (audio, target) in tqdm(enumerate(loader), total=len(loader), desc="Evaluate", disable=True):
        with torch.no_grad():
            embeddings = model(audio.to(device))
            compatibility = comp_func(embeddings, class_embeddings)

        predictions.append(compatibility.argmax(dim=1).cpu().numpy())
        targets.append(target.numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    results = {key: metrics[key](targets, predictions) for key in metrics}
    return results, predictions, targets


def main(cfg):
    set_random_seed(cfg.hparams.seed)

    label = 'species'
    embeddings = 'class_embeddings'
    filename = 'filename'

    device = cfg.meta.device

    results_folder = cfg.meta.results_root
    os.makedirs(results_folder, exist_ok=True)

    results_list = []
    metrics = get_metrics()

    for fold in tqdm(range(5)):
        fold_results = os.path.join(results_folder, str(fold))

        audio = pd.read_csv(cfg.meta.audio_features).dropna()

        feature_source = get_meta_source(cfg.meta.meta_type, cfg)

        meta = pd.read_csv(feature_source)

        audio[label] = audio[filename].apply(lambda x: x.split('/')[0])
        audio[filename] = audio[filename].apply(lambda x: x.split('/')[1])

        min_max_scaler = preprocessing.MinMaxScaler()
        meta.iloc[:, 1:] = min_max_scaler.fit_transform(meta.iloc[:, 1:])

        feature_names = list(set(audio.columns) - {"filename", label})

        audio[feature_names] = audio[feature_names].astype(float)

        split_type = cfg.meta.split
        split_func = get_splitting_function(split_type)

        params = ((list(audio[label].unique()))) if split_type == 'random' else (
            cfg.meta.predefined_zsl_folds, fold, label
        )

        train_species, dev_species, test_species = split_func(*params)

        species_lists = {"train": train_species, "dev": dev_species, "test": test_species}

        partitions = {}
        standardizer = None

        for split in species_lists.keys():
            split_audio = audio.loc[audio[label].isin(species_lists[split])].reset_index()
            split_meta = meta.loc[meta[label].isin(species_lists[split])]

            encoder = LabelEncoder(split_audio[label].unique())
            split_audio[label] = split_audio[label].apply(encoder.encode)
            split_meta[label] = split_meta[label].apply(encoder.encode)

            encoder.to_yaml(os.path.join(fold_results, f"encoder.{split}.yaml"))

            split_meta = split_meta.sort_values(by=label)

            class_embeddings = torch.from_numpy(split_meta[list(set(split_meta.columns) - {label})]
                                                .values).float().to(device)

            if split == "train":
                values = split_audio[feature_names].values

                standardizer = Standardizer(values.mean(axis=0), values.std(axis=0))
                standardizer.to_yaml(os.path.join(fold_results, f"audio.scaler.yaml"))

                assert standardizer is not None

            dataset = Dataset(audio=split_audio[feature_names + [label]], transform=standardizer)

            partitions[split] = {"dataset": dataset, embeddings: class_embeddings}

        writer = SummaryWriter(log_dir=os.path.join(fold_results, 'log'))

        model = torch.nn.Linear(len(feature_names), len(set(meta.columns) - {label}), bias=False)

        batch_size = cfg.hparams.batch_size

        train_loader = DataLoader(dataset=partitions["train"]["dataset"], batch_size=batch_size)
        dev_loader = DataLoader(dataset=partitions["dev"]["dataset"], batch_size=batch_size)
        test_loader = DataLoader(dataset=partitions["test"]["dataset"], batch_size=batch_size)

        hyperparams = cfg.hparams
        optimizer_name, lr = hyperparams.optimizer, hyperparams.learning_rate
        optimizer = get_optimizer(optimizer_name, lr, list(model.parameters()))

        loss_func = get_loss_function(hyperparams.loss)
        comp_func = get_compatibility_function(hyperparams.compatibility)
        epochs = hyperparams.epochs

        best_epoch, best_state, best_results = None, None, None
        best_metric = 0

        if not os.path.exists(os.path.join(fold_results, 'best.pth.tar')):
            for epoch in range(epochs):
                train_epoch(
                    loader=train_loader, model=model, class_emb=partitions["train"][embeddings],
                    optimizer=optimizer, comp_func=comp_func, loss_func=loss_func, device=device,
                    writer=writer, epoch=epoch
                )

                results, predictions, targets = evaluate(
                    loader=dev_loader, model=model, class_embeddings=partitions["dev"][embeddings],
                    comp_func=comp_func, device=device, metrics=metrics
                )

                print(f"Dev results at epoch {epoch + 1}:")
                print(yaml.dump(results))

                torch.save(model.cpu().state_dict(), os.path.join(fold_results, 'last.pth.tar'))

                for key in results.keys():
                    writer.add_scalar(f'Dev/{key}', results[key], epoch)
                if results["F1"] > best_metric:
                    best_metric = results["F1"]
                    best_results = results.copy()
                    best_state = model.cpu().state_dict()
                    best_epoch = epoch + 1
                    torch.save(best_state, os.path.join(fold_results, 'best.pth.tar'))

            best_results["Epoch"] = best_epoch
            writer.close()
            with open(os.path.join(fold_results, 'dev.yaml'), 'w') as fp:
                yaml.dump(best_results, fp)

            print(f"Best results found in epoch: {best_epoch}.")
        else:
            best_state = torch.load(os.path.join(fold_results, 'best.pth.tar'))

        model.load_state_dict(best_state)

        results, predictions, targets = evaluate(
            loader=test_loader, model=model, class_embeddings=partitions["test"][embeddings],
            comp_func=comp_func, device=device, metrics=metrics
        )

        print(f"Test results ({len(encoder.labels)} classes):")
        print(yaml.dump(results))
        with open(os.path.join(fold_results, 'test.yaml'), 'w') as fp:
            yaml.dump(results, fp)

        results_list.append(results)

    for i, result in enumerate(results_list):
        print(f"\nFold {i} ", result)

    scores = {}
    for metric in metrics.keys():
        name = f"Mean {metric}"
        scores[name] = 0
        for result in results_list:
            scores[name] += result[metric]
        scores[name] = scores[name] / len(results_list)

    print('\n\nResults averaged per fold (test)\n' + yaml.dump(scores, allow_unicode=True, default_flow_style=False))

    with open(os.path.join(results_folder, 'total_results.yaml'), 'w') as fp:
        yaml.dump(scores, fp)
