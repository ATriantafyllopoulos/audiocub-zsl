import os
import sys
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder


if __name__=='__main__':
    mapping = {
        'carduelis cannabina': 'linaria cannabina', 
        'carduelis chloris': 'chloris chloris',
        'carduelis spinus': 'spinus spinus',
        'corvus corone cornix': 'corvus cornix',
        'corvus corone corone': 'corvus corone',
        'dendrocopos medius': 'leiopicus medius', # Synonyms (in history dataset Leiopicus, in AVONET dendrocopos)
        'dendrocopos minor': 'dryobates minor', # Synonym
        'parus cristatus': 'lophophanes cristatus',
        'parus montanus': 'poecile montanus',
        'parus palustris': 'poecile palustris',
        'regulus ignicapillus': 'regulus ignicapilla',
        'sylvia curucca': 'sylvia curruca',
        'saxicola rubicola': 'saxicola torquatus'
    }

    # with open("/nas/staff/data_work/AG/BIRDS/baseline_embeddings/mapping.json", "w") as f:
    #     json.dump(mapping, f) 

    ####################
    # HISTORY DRYAD (BLH)
    ####################
    path = "/home/gebhaale/Downloads/doi_10.5061_dryad.n6k3n__v1/Life-history characteristics of European birds.txt"
    df = pd.read_csv(path, sep="\t", encoding="latin1")
    df.rename(columns={'Species': 'species'}, inplace=True)
    df.dropna(subset=['species'], inplace=True)
    df['species'] = df['species'].apply(lambda x: x.lower())

    df_hist = df.iloc[:, 3:]
    df_hist['species'] = [mapping[x] if x in mapping.keys() else x for x in df_hist['species']]
    print(df_hist.shape)


    birds_species = os.listdir("/nas/staff/data_work/AG/BIRDS/datasets")
    birds_species = [x.lower().replace("_", " ") for x in birds_species]
    birds_species_mapped = [mapping[x] if x in mapping.keys() else x for x in birds_species]


    df_hist = df_hist[df_hist['species'].isin(birds_species_mapped)]
    row_to_duplicate = df_hist.loc[df_hist['species'] == 'corvus corone'].copy()
    row_to_duplicate['species'] = 'corvus cornix'
    df_hist = pd.concat([df_hist, row_to_duplicate], ignore_index=True)

    columns_with_nan = df_hist.columns[df_hist.isna().sum() >= 10].values.tolist()
    columns_with_nan.append("Data source")
    df_hist.drop(columns=columns_with_nan, inplace=True)
    print(df_hist.shape)


    tmp = df_hist.fillna(0)
    for column in tmp.iloc[:, 1:].select_dtypes(include='object'):
        print("Column: ", column)
        tmp[column] = LabelEncoder().fit_transform(tmp[column].apply(str))

    tmp.to_csv("/nas/staff/data_work/AG/BIRDS/baseline_embeddings/history_embeddings.csv", index=None)


    ####################
    # AVONET
    ####################
    path_avo = "/home/gebhaale/Downloads/AVONET/AVONET Supplementary dataset 1.xlsx"
    avo_df = pd.read_excel(path_avo, sheet_name="AVONET1_BirdLife")
    avo_df = avo_df.iloc[:, :-5]
    avo_df['species'] = [x.lower() for x in avo_df["Species1"]] 
    avo_df = avo_df.iloc[:, 10:]
    print(avo_df.shape)
    print(avo_df.head(5))


    birds_species = os.listdir("/nas/staff/data_work/AG/BIRDS/datasets")
    birds_species = [x.lower().replace("_", " ") for x in birds_species]
    birds_species_mapped = [mapping[x] if x in mapping.keys() else x for x in birds_species]

    row_to_duplicate = avo_df.loc[avo_df['species'] == 'corvus corone'].copy()
    row_to_duplicate['species'] = 'corvus cornix'
    avo_df = pd.concat([avo_df, row_to_duplicate], ignore_index=True)

    # Move the last column to the front
    last_column = avo_df.pop(avo_df.columns[-1])
    avo_df.insert(0, last_column.name, last_column)
    # Filter for our species
    avo_df = avo_df[avo_df['species'].isin(birds_species_mapped)]
    print(avo_df.head(5))

    columns_with_nan = avo_df.columns[avo_df.isna().sum() >= 10].values.tolist()
    avo_df.drop(columns=columns_with_nan, inplace=True)
    print(avo_df.shape)


    tmp = avo_df.fillna(0)
    for column in tmp.iloc[:, 1:].select_dtypes(include='object'):
        print("Column: ", column)
        tmp[column] = LabelEncoder().fit_transform(tmp[column].apply(str))

    tmp.to_csv("/nas/staff/data_work/AG/BIRDS/baseline_embeddings/avonet_embeddings.csv", index=None)