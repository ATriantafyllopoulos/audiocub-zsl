import audmetric
import audtorch
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
    Dataset2D,
    LabelEncoder,
    Standardizer,
    random_split,
    load_split_for_fold
)
from loss_own import (
    ranking_loss,
    devise_loss,
    ranking_loss_UNCol,
    ranking_loss_UNRow
)

from models import (
    TransformerClassifier
)

from sklearn import (
    preprocessing
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
    embeddings_are_2D = cfg.meta.embeddings_are_2D
    use_predefined_clf_sets = cfg.meta.use_predefined_clf_sets
    fold_dir = cfg.meta.predefined_clf_splits

    # set random seeds for reproducibility
    torch.manual_seed(cfg.hparams.seed)
    np.random.seed(cfg.hparams.seed)
    random.seed(cfg.hparams.seed)

    experiment_folder = cfg.meta.results_root
    os.makedirs(experiment_folder, exist_ok=True)

    normalisation_type = cfg.hparams.normalisation_type

    results_list = []
    # folds = [1]
    for fold_id in range(5):
        print(f"--------FOLD {fold_id}------------")
        experiment_folder = os.path.join(experiment_folder, str(fold_id))

        ###############################################
        # Load data
        ###############################################

        # Load mapping dictionary
        with open("/nas/staff/data_work/AG/BIRDS/baseline_embeddings/mapping.json", "r") as f:
            mapping = json.load(f)

        # Check if pre-defined zsl splits shall be used and load them if that's the case
        if use_predefined_clf_sets:
            # Load the overall ast data
            audio = pd.read_csv(cfg.meta.audio_features).dropna()
            if 'file' in audio.columns:
                audio = audio.rename(columns={'file': 'filename'})
            audio['filename'] = audio['filename'].apply(lambda x: x.lstrip('/'))
            audio["species"] = audio["filename"].apply(lambda x: x.split('/')[0])

            # Map the species names
            audio["species"] = audio["species"].apply(lambda x: x.lower().replace("_", " "))
            audio["species"] = [mapping[s] if s in mapping.keys() else s for s in audio["species"]]

            # load train, dev, and test set regarding the splits
            train_df = pd.read_csv(os.path.join(fold_dir, 'train.csv')).dropna()
            train_df['filename'] = train_df['filename'].apply(lambda x: x.lstrip('/'))
            if 'annotation' in train_df.columns:
                train_df = train_df.rename(columns={'annotation': 'species'})

            dev_df = pd.read_csv(os.path.join(fold_dir, 'devel.csv')).dropna()
            dev_df['filename'] = dev_df['filename'].apply(lambda x: x.lstrip('/'))
            if 'annotation' in dev_df.columns:
                dev_df = dev_df.rename(columns={'annotation': 'species'})

            test_df = pd.read_csv(os.path.join(fold_dir, 'test.csv')).dropna()
            test_df['filename'] = test_df['filename'].apply(lambda x: x.lstrip('/'))
            if 'annotation' in test_df.columns:
                test_df = test_df.rename(columns={'annotation': 'species'})

            # Map the species names
            train_df["species"] = train_df["species"].apply(lambda x: x.lower().replace("_", " "))
            train_df["species"] = [mapping[s] if s in mapping.keys() else s for s in train_df["species"]]
            dev_df["species"] = dev_df["species"].apply(lambda x: x.lower().replace("_", " "))
            dev_df["species"] = [mapping[s] if s in mapping.keys() else s for s in dev_df["species"]]
            test_df["species"] = test_df["species"].apply(lambda x: x.lower().replace("_", " "))
            test_df["species"] = [mapping[s] if s in mapping.keys() else s for s in test_df["species"]]
            total_species = pd.concat([train_df, dev_df, test_df], ignore_index=True)["species"].unique()
            print("Total number of species: ", len(total_species))

            # Load the text features
            text = pd.read_csv(cfg.meta.text_features)
            # Map the species names
            text["species"] = text["species"].apply(lambda x: x.lower().replace("_", " "))
            text["species"] = [mapping[s] if s in mapping.keys() else s for s in text["species"]]
            text = text.loc[text["species"].isin(total_species)]


            # MinMax standardise between 0 and 1 if avonet/dryad features
            if 'concatenated' not in cfg.meta.text_features.lower() and ('avonet' in cfg.meta.text_features.lower() or 'history' in cfg.meta.text_features.lower()):
                # min_values = text.iloc[:, 1:].min(axis=1)
                # max_values = text.iloc[:, 1:].max(axis=1)
                # text.iloc[:, 1:] = (text.iloc[:, 1:] - min_values[:, None]) / (max_values[:, None] - min_values[:, None])
                # column normalisation
                if normalisation_type == "column":
                    min_max_scaler = preprocessing.MinMaxScaler()
                    text.iloc[:, 1:] = min_max_scaler.fit_transform(text.iloc[:, 1:])
                # row normalisation
                elif normalisation_type == "row":
                    min_values = text.iloc[:, 1:].min(axis=1)
                    max_values = text.iloc[:, 1:].max(axis=1)
                    text.iloc[:, 1:] = (text.iloc[:, 1:] - min_values[:, None]) / (max_values[:, None] - min_values[:, None])
            
            if not embeddings_are_2D:
                feature_names = list(set(audio.columns) - set(["filename", "species"]))
                # print("Feature names: ", feature_names)
                # audio = audio.drop(["filename"], axis=1) # TODO: maybe drop anyways?
                for col in feature_names:
                    audio[col] = audio[col].astype(float)
            else:
                audio['features'] = audio['features'].apply(lambda x: os.path.join(os.path.dirname(cfg.meta.audio_features), x))

        else:
            # Load the data
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

            # MinMax standardise between 0 and 1 if avonet/dryad features
            if 'concatenated' not in cfg.meta.text_features.lower() and ('avonet' in cfg.meta.text_features.lower() or 'history' in cfg.meta.text_features.lower()):
            # if 'avonet' in cfg.meta.text_features.lower() or 'history' in cfg.meta.text_features.lower() or 'concatenated' in cfg.meta.text_features.lower():
                # # row normalisation
                # min_values = text.iloc[:, 1:].min(axis=1)
                # max_values = text.iloc[:, 1:].max(axis=1)
                # text.iloc[:, 1:] = (text.iloc[:, 1:] - min_values[:, None]) / (max_values[:, None] - min_values[:, None])
                # column normalisa# Check if pre-defined zsl splits shall be used and load them if that's the casetion
                if normalisation_type == "column":
                    min_max_scaler = preprocessing.MinMaxScaler()
                    text.iloc[:, 1:] = min_max_scaler.fit_transform(text.iloc[:, 1:])
                # row normalisation
                elif normalisation_type == "row":
                    min_values = text.iloc[:, 1:].min(axis=1)
                    max_values = text.iloc[:, 1:].max(axis=1)
                    text.iloc[:, 1:] = (text.iloc[:, 1:] - min_values[:, None]) / (max_values[:, None] - min_values[:, None])

            if not embeddings_are_2D:
                feature_names = list(set(audio.columns) - set(["filename", "species"]))
                # print("Feature names: ", feature_names)
                # audio = audio.drop(["filename"], axis=1) # TODO: maybe drop anyways?
                for col in feature_names:
                    audio[col] = audio[col].astype(float)
            else:
                audio['features'] = audio['features'].apply(lambda x: os.path.join(os.path.dirname(cfg.meta.audio_features), x))

        ###############################################
        # Create splits
        ###############################################
        if cfg.meta.split == "random":
            splitting_function = random_split
        elif cfg.meta.split == "predefined_zsl_folds":
            splitting_function = load_split_for_fold
        else:
            raise NotImplementedError(cfg.meta.split)


        # Check if pre-defined zsl splits shall be used and load them if that's the case
        if use_predefined_clf_sets:
            # filter the filenames and species of each split
            train_filenames, dev_filenames, test_filenames = train_df["filename"], dev_df["filename"], test_df["filename"]
            filenames_lists = {
                "train": train_filenames,
                "dev": dev_filenames,
                "test": test_filenames
            }
            train_species, dev_species, test_species = list(train_df["species"].unique()), list(dev_df["species"].unique()), list(test_df["species"].unique())
            species_lists = {
                "train": train_species,
                "dev": dev_species,
                "test": test_species
            }

            partitions = {}
            standardizer = None
            for split in ["train", "dev", "test"]:
                split_audio = audio.loc[audio["filename"].isin(filenames_lists[split])].reset_index()
                split_text = text.loc[text["species"].isin(species_lists[split])]

                # encode classes 
                encoder = LabelEncoder(split_audio["species"].unique())
                split_audio["species"] = split_audio["species"].apply(encoder.encode)
                split_text["species"] = split_text["species"].apply(encoder.encode)
                encoder.to_yaml(os.path.join(experiment_folder, f"encoder.{split}.yaml"))

                # get class embeddings
                split_text = split_text.sort_values(by="species")
                class_embeddings = torch.from_numpy(
                    split_text[list(set(split_text.columns) - set(["species"]))].values
                ).float().to(cfg.meta.device)

                if not embeddings_are_2D:
                    split_audio = split_audio.drop(["filename"], axis=1)
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
                else:
                    if split != "test":
                        partitions[split] = {
                            "dataset": Dataset2D(audio=split_audio, transform=audtorch.transforms.RandomCrop(1214, axis=-2)),
                            "class_embeddings": class_embeddings
                        }
                    else:
                        partitions[split] = {
                            "dataset": Dataset2D(audio=split_audio, transform=audtorch.transforms.Expand(1214, axis=-2)),
                            "class_embeddings": class_embeddings
                        }

        else:
            if cfg.meta.split == "predefined_zsl_folds":
                train_species, dev_species, test_species = splitting_function(mapping=mapping, fold_idx=fold_id)
            else:
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
                if not embeddings_are_2D:
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
                else:
                    if split != "test":
                        partitions[split] = {
                            "dataset": Dataset2D(audio=split_audio, transform=audtorch.transforms.RandomCrop(1214, axis=-2)),
                            "class_embeddings": class_embeddings
                        }
                    else:
                        partitions[split] = {
                            "dataset": Dataset2D(audio=split_audio, transform=audtorch.transforms.Expand(1214, axis=-2)),
                            "class_embeddings": class_embeddings
                        }

        writer = SummaryWriter(
            log_dir=os.path.join(experiment_folder, 'log')
        )

        if cfg.meta.model == 'transformer':
            model = TransformerClassifier(
                d_model=len(feature_names) if not embeddings_are_2D else len(set(text.columns) - set(["species"])), 
                output_dim=len(set(text.columns) - set(["species"])),
                ff_hidden_size=len(feature_names) if not embeddings_are_2D else len(set(text.columns) - set(["species"])),
                embeddings_are_2D=embeddings_are_2D
            )
        else:
            # TODO: handle 2D audio embeddings
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
        elif cfg.hparams.loss == "ranking_UNCol":
            loss_function = ranking_loss_UNCol
        elif cfg.hparams.loss == "ranking_UNRow":
            loss_function = ranking_loss_UNRow
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
            # compatibility_function=dot_product_compatibility, # always dot_product_compatibility for testing...why?
            compatibility_function=compatibility_function,
            device=cfg.meta.device
        )
        print(f"Test results ({len(encoder.labels)} classes):")
        print(yaml.dump(results))
        with open(os.path.join(experiment_folder, 'test.yaml'), 'w') as fp:
            yaml.dump(results, fp)
        
        results_list.append(results)
    

    for i, result in enumerate(results_list):
        print(f"\nFold {i} ", result)

    uar, acc, f_score = 0, 0, 0
    for result in results_list:
        uar += result['UAR']
        acc += result['ACC']
        f_score += result['F1']
    
    print("\nMean UAR: ", uar / len(results_list))
    print("Mean ACC: ", acc / len(results_list))
    print("Mean F1: ", f_score / len(results_list))
