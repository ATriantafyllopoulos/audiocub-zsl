import os
import sys
import pandas as pd
import json
import argparse
from sklearn.preprocessing import LabelEncoder


if __name__=='__main__':
    parser = argparse.ArgumentParser("Process meta information to appropriate embeddings")
    parser.add_argument(
        "--audiodir", 
        type=str, 
        required=True, 
        help="The directory path which contains the bird species folders with the audio recordings"
    )
    parser.add_argument(
        "--meta-information",
        type=str, 
        required=True, 
        help="The filepath of the CSV containing the meta information"
    )
    parser.add_argument(
        "--storepath",
        type=str, 
        required=True, 
        help="The directory in which the processed meta information shall be stored"
    )
    parser.add_argument(
        "--mapping-path",
        type=str, 
        required=True, 
        help="The filepath of the mapping json file for the bird species synonyms"
    )
    parser.add_argument(
        "--meta-type",
        type=str, 
        required=True,
        choices=[
            'avonet',
            'blh'
        ],
        help="The kind of meta information which shall be processed. Allows only the functional traits."
    )
    args = parser.parse_args()
    print("Arguments:\n", args)
    os.makedirs(args.storepath, exist_ok=True)
    audiodir = args.audiodir

    # Load mapping dictionary
    with open(args.mapping_path, "r") as f:
        mapping = json.load(f)

    if args.meta_type == 'blh':
        ####################
        # Bird Life History from DRYAD (BLH)
        ####################
        path = args.meta_information # e.g. ./meta_information/blh.txt
        df = pd.read_csv(path, sep="\t", encoding="latin1")
        df.rename(columns={'Species': 'species'}, inplace=True)
        df.dropna(subset=['species'], inplace=True)
        df['species'] = df['species'].apply(lambda x: x.lower())

        df_hist = df.iloc[:, 3:]
        df_hist['species'] = [mapping[x] if x in mapping.keys() else x for x in df_hist['species']]
        print(df_hist.shape)


        birds_species = os.listdir(audiodir)
        birds_species = [x.lower().replace("_", " ") for x in birds_species]
        birds_species_mapped = [mapping[x] if x in mapping.keys() else x for x in birds_species]

        # corvus corone and corvus cornix seem to be quite similar, so we duplicate and name the duplicated row corvus cornix (needed as it occurs in the meta info)
        df_hist = df_hist[df_hist['species'].isin(birds_species_mapped)]
        row_to_duplicate = df_hist.loc[df_hist['species'] == 'corvus corone'].copy()
        row_to_duplicate['species'] = 'corvus cornix'
        df_hist = pd.concat([df_hist, row_to_duplicate], ignore_index=True)

        columns_with_nan = df_hist.columns[df_hist.isna().sum() >= 10].values.tolist()
        columns_with_nan.append("Data source")
        df_hist.drop(columns=columns_with_nan, inplace=True)
        tmp = df_hist.fillna(0)

    else:
        ####################
        # AVONET
        ####################
        path_avo = args.meta_information # e.g. ./meta_information/avonet.xlsx
        avo_df = pd.read_excel(path_avo, sheet_name="AVONET1_BirdLife")
        avo_df = avo_df.iloc[:, :-5]
        avo_df['species'] = [x.lower() for x in avo_df["Species1"]] 
        avo_df = avo_df.iloc[:, 10:]
        print(avo_df.shape)
        print(avo_df.head(5))


        birds_species = os.listdir(audiodir)
        birds_species = [x.lower().replace("_", " ") for x in birds_species]
        birds_species_mapped = [mapping[x] if x in mapping.keys() else x for x in birds_species]

        # corvus corone and corvus cornix seem to be quite similar, so we duplicate and name the duplicated row corvus cornix (needed as it occurs in the meta info)
        row_to_duplicate = avo_df.loc[avo_df['species'] == 'corvus corone'].copy() 
        row_to_duplicate['species'] = 'corvus cornix'
        avo_df = pd.concat([avo_df, row_to_duplicate], ignore_index=True)

        # Move the last column to the front
        last_column = avo_df.pop(avo_df.columns[-1])
        avo_df.insert(0, last_column.name, last_column)
        # Filter for our species
        avo_df = avo_df[avo_df['species'].isin(birds_species_mapped)]

        columns_with_nan = avo_df.columns[avo_df.isna().sum() >= 10].values.tolist()
        avo_df.drop(columns=columns_with_nan, inplace=True)
        tmp = avo_df.fillna(0)

    for column in tmp.iloc[:, 1:].select_dtypes(include='object'):
        print("Column: ", column)
        tmp[column] = LabelEncoder().fit_transform(tmp[column].apply(str))

    tmp.to_csv(f"{args.storepath}/{args.meta_type}_embeddings.csv", index=None)