import audmetric
import numpy as np
import os
import pandas as pd
import random
import torch
import tqdm
import typing
import yaml
import json

from torch.utils.tensorboard import SummaryWriter

from compatibility import (
    dot_product_compatibility,
    euclidean_distance_compatibility,
    cosine_similarity_compatibility,
    manhattan_distance_compatibility
)
from data import (
    Dataset,
    LabelEncoder,
    Standardizer,
    random_split
)
from loss_own import (
    ranking_loss,
    devise_loss
)

from models import (
    TransformerClassifier
)


def train_epoch(
    loader,
    model,
    class_embeddings,
    optimizer,
    device,
    compatibility_function,
    loss_function,
    writer,
    epoch
):
    model.train()
    model.to(device)
    for index, (audio, targets) in tqdm.tqdm(
        enumerate(loader),
        total=len(loader),
        desc="Training",
        disable=True
    ):
        embeddings = model(audio.to(device))
        
        total_loss = loss_function(
            embeddings=embeddings, 
            class_embeddings=class_embeddings, 
            targets=targets,
            compatibility_function=compatibility_function,
            model=model
        )
        # loss = total_loss.sum() # why sum and not mean?
        loss = total_loss.mean() 
        # print("Loss mean: ", loss)

        if index % 50 == 0:
            writer.add_scalar(
                'Loss',
                loss,
                epoch * len(loader) + index
            )


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # import sys
        # sys.exit()


def evaluate(
    loader,
    model,
    class_embeddings,
    device,
    compatibility_function
):
    metrics = {
        "ACC": audmetric.accuracy,
        "UAR": audmetric.unweighted_average_recall,
        "F1": audmetric.unweighted_average_fscore
    }
    model.eval()
    model.to(device)
    predictions = []
    targets = []
    for index, (audio, target) in tqdm.tqdm(
        enumerate(loader),
        total=len(loader),
        desc="Training",
        disable=True
    ):
        with torch.no_grad():
            embeddings = model(audio.to(device))
            compatibility = compatibility_function(embeddings, class_embeddings)
        predictions.append(compatibility.argmax(dim=1).cpu().numpy())
        targets.append(target.numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    results = {
        key: metrics[key](targets, predictions)
        for key in metrics
    }
    return results, predictions, targets


def main(cfg):

    # set random seeds for reproducibility
    torch.manual_seed(cfg.hparams.seed)
    np.random.seed(cfg.hparams.seed)
    random.seed(cfg.hparams.seed)

    experiment_folder = cfg.meta.results_root
    os.makedirs(experiment_folder, exist_ok=True)

    ###############################################
    # Load data
    ###############################################

    # Load mapping dictionary
    with open("/nas/staff/data_work/AG/BIRDS/baseline_embeddings/mapping.json", "r") as f:
        mapping = json.load(f)

    audio = pd.read_csv(cfg.meta.audio_features).dropna()
    text = pd.read_csv(cfg.meta.text_features)
    if 'file' in audio.columns:
        audio = audio.rename(columns={'file': 'filename'})
    audio['filename'] = audio['filename'].apply(lambda x: x.lstrip('/'))
    audio["species"] = audio["filename"].apply(lambda x: x.split('/')[0])

    # Map the species names
    audio["species"] = audio["species"].apply(lambda x: x.lower().replace("_", " "))
    audio["species"] = [mapping[s] if s in mapping.keys() else s for s in audio["species"]]
    text["species"] = text["species"].apply(lambda x: x.lower().replace("_", " "))
    text["species"] = [mapping[s] if s in mapping.keys() else s for s in text["species"]]

    text = text.loc[text["species"].isin(audio["species"].unique())]
    feature_names = list(set(audio.columns) - set(["filename", "species"]))
    # print("Feature names: ", feature_names)
    audio = audio.drop(["filename"], axis=1)
    for col in feature_names:
        audio[col] = audio[col].astype(float)

    ###############################################
    # Create splits
    ###############################################
    if cfg.meta.split == "random":
        splitting_function = random_split
    else:
        raise NotImplementedError(cfg.meta.split)

    train_species, dev_species, test_species = splitting_function(list(audio["species"].unique()))
    species_lists = {
        "train": train_species,
        "dev": dev_species,
        "test": test_species
    }
    partitions = {}
    standardizer = None
    for split in ["train", "dev", "test"]:
        # select appropriate rows
        split_audio = audio.loc[audio["species"].isin(species_lists[split])].reset_index()
        split_text = text.loc[text["species"].isin(species_lists[split])]
        
        # one-hot encode classes # but it's not one-hot encoded, is it?
        encoder = LabelEncoder(split_audio["species"].unique())
        split_audio["species"] = split_audio["species"].apply(encoder.encode)
        split_text["species"] = split_text["species"].apply(encoder.encode)
        encoder.to_yaml(os.path.join(experiment_folder, f"encoder.{split}.yaml"))

        # get class embeddings
        split_text = split_text.sort_values(by="species")
        class_embeddings = torch.from_numpy(
            split_text[list(set(split_text.columns) - set(["species"]))].values
        ).float().to(cfg.meta.device)

        # create feature normalization
        # TODO: makes sense to allow for different options using hydra
        # TODO: can use text standardizer as well
        if split == "train":
            standardizer = Standardizer(
                split_audio[feature_names].values.mean(axis=0),
                split_audio[feature_names].values.std(axis=0)
            )
            standardizer.to_yaml(os.path.join(experiment_folder, f"audio.scaler.yaml"))
        assert standardizer is not None

        partitions[split] = {
            "dataset": Dataset(audio=split_audio[feature_names + ["species"]], transform=standardizer),
            "class_embeddings": class_embeddings
        }

    writer = SummaryWriter(
        log_dir=os.path.join(experiment_folder, 'log')
    )

    if cfg.meta.model == 'transformer':
        model = TransformerClassifier(
            d_model=len(feature_names), 
            output_dim=len(set(text.columns) - set(["species"])),
            ff_hidden_size=len(feature_names)
        )
    else:
        model = torch.nn.Linear(
            len(feature_names),
            len(set(text.columns) - set(["species"])),
            bias=False
        )
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset=partitions["train"]["dataset"],
        shuffle=True,
        batch_size=cfg.hparams.batch_size
    )
    dev_loader = torch.utils.data.DataLoader(
        dataset=partitions["dev"]["dataset"],
        batch_size=cfg.hparams.batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=partitions["test"]["dataset"],
        batch_size=cfg.hparams.batch_size
    )

    if cfg.hparams.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            momentum=0.9, 
            lr=cfg.hparams.learning_rate
        )
    elif cfg.hparams.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.hparams.learning_rate
        )
    elif cfg.hparams.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.hparams.learning_rate,
            weight_decay=0.0001
        )
    elif cfg.hparams.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=cfg.hparams.learning_rate,
            alpha=.95,
            eps=1e-7
        )
    else:
        raise NotImplementedError(cfg.hparams.optimizer)
    epochs = cfg.hparams.epochs

    if cfg.hparams.loss == "ranking":
        loss_function = ranking_loss
    elif cfg.hparams.loss == "devise":
        loss_function = devise_loss
    else:
        raise NotImplementedError(cfg.hparams.loss)
    
    if cfg.hparams.compatibility == "dot_product":
        compatibility_function = dot_product_compatibility
    elif cfg.hparams.compatibility == "euclidean_distance":
        compatibility_function = euclidean_distance_compatibility
    elif cfg.hparams.compatibility == "manhattan_distance":
        compatibility_function = manhattan_distance_compatibility
    elif cfg.hparams.compatibility == "cosine_similarity":
        compatibility_function = cosine_similarity_compatibility
    else:
        raise NotImplementedError(cfg.hparams.compatibility)
    best_epoch = None
    best_metric = 0
    best_state = None
    best_results = None
    if not os.path.exists(os.path.join(experiment_folder, 'best.pth.tar')):
        for epoch in range(epochs):
            train_epoch(
                loader=train_loader,
                model=model,
                class_embeddings=partitions["train"]["class_embeddings"],
                optimizer=optimizer,
                compatibility_function=compatibility_function,
                loss_function=loss_function,
                device=cfg.meta.device,
                writer=writer,
                epoch=epoch
            )
            results, predictions, targets = evaluate(
                loader=dev_loader,
                model=model,
                class_embeddings=partitions["dev"]["class_embeddings"],
                compatibility_function=compatibility_function,
                device=cfg.meta.device
            )
            print(f"Dev results at epoch {epoch+1}:")
            print(yaml.dump(results))
            torch.save(model.cpu().state_dict(), os.path.join(
                experiment_folder, 'last.pth.tar'))
            for key in results.keys():
                writer.add_scalar(f'Dev/{key}', results[key], epoch)
            if results["F1"] > best_metric:
                best_metric = results["F1"]
                best_results = results.copy()
                best_state = model.cpu().state_dict()
                best_epoch = epoch + 1
                torch.save(best_state, os.path.join(
                    experiment_folder, 'best.pth.tar'))
        best_results["Epoch"] = best_epoch
        writer.close()
        with open(os.path.join(experiment_folder, 'dev.yaml'), 'w') as fp:
            yaml.dump(best_results, fp)
        print(f"Best results found in epoch: {best_epoch}.")
    else:
        best_state = torch.load(os.path.join(experiment_folder, 'best.pth.tar'))
    model.load_state_dict(best_state)
    results, predictions, targets = evaluate(
        loader=test_loader,
        model=model,
        class_embeddings=partitions["test"]["class_embeddings"],
        compatibility_function=dot_product_compatibility,
        device=cfg.meta.device
    )
    print(f"Test results ({len(encoder.labels)} classes):")
    print(yaml.dump(results))
    with open(os.path.join(experiment_folder, 'test.yaml'), 'w') as fp:
        yaml.dump(results, fp)
