import os
import argparse
from sklearn import  preprocessing

if __name__=='__main__':
    parser = argparse.ArgumentParser("Concatenate embeddings")
    parser.add_argument(
        "--bert",
        type=str,
        required=True,
        help="Filepath to BERT embeddings"
    )
    parser.add_argument(
        "--avonet",
        type=str,
        required=True,
        help="Filepath to avonet embeddings"
    )
    parser.add_argument(
        "--blh",
        type=str,
        required=True,
        help="Filepath to BLH embeddings"
    )
    parser.add_argument(
        "--storepath",
        type=str,
        required=True,
        help="Directory to store the concatenated embeddings"
    )
    args = parser.parse_args()
    print("Arguments:\n", args)

    bert_embeddings = pd.read_csv(args.bert)
    avonet_embeddings = pd.read_csv(args.avonet)
    blh_embeddings = pd.read_csv(args.blh)

    # Normalise avonet and blh column-wise
    min_max_scaler = preprocessing.MinMaxScaler()
    avonet_embeddings.iloc[:, 1:] = min_max_scaler.fit_transform(avonet_embeddings.iloc[:, 1:])
    min_max_scaler = preprocessing.MinMaxScaler()
    blh_embeddings.iloc[:, 1:] = min_max_scaler.fit_transform(blh_embeddings.iloc[:, 1:])

    # Concatenate the embeddings
    df_concat = pd.concat([bert_embeddings, avonet_embeddings.drop(columns=['species']), blh_embeddings.drop(columns=['species'])], axis=1)
    df_concat_bertavonet = pd.concat([bert_embeddings, avonet_embeddings.drop(columns=['species'])], axis=1)
    df_concat_bertblh = pd.concat([bert_embeddings, blh_embeddings.drop(columns=['species'])], axis=1)
    df_concat_avonetblh = pd.concat([avonet_embeddings, blh_embeddings.drop(columns=['species'])], axis=1)


    # Store the concatenated embeddings
    df_concat.to_csv(os.path.join(args.storepath, 'normalisedColumns_concatenated_embeddings.csv'), index=None)
    df_concat_bertavonet.to_csv(os.path.join(args.storepath, 'normalisedColumns_concatenated_bertavonet_embeddings.csv'), index=None)
    df_concat_bertblh.to_csv(os.path.join(args.storepath, 'normalisedColumns_concatenated_bertblh_embeddings.csv'), index=None)
    df_concat_avonetblh.to_csv(os.path.join(args.storepath, 'normalisedColumns_concatenated_avonetblh_embeddings.csv'), index=None)