import torch
import os
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

if __name__=='__main__':
    parser = argparse.ArgumentParser("(S)BERT similarity")
    parser.add_argument(
        "--bert-embeddings",
        type=str,
        help="The filepath of the (S)BERT embedding CSV"
    )
    parser.add_argument(
        "--embeddings_type",
        type=str,
        default="bert",
        choices=[
            "bert",
            "sbert"
        ],
        help="The filepath of the SBERT embedding CSV"
    )
    args = parser.parse_args()
    print("Arguments:\n", args)

    bert_embeddings = pd.read_csv(args.bert_embeddings)

    # Calculate and plot similarity measures among the species
    pairwise_distances = cdist(bert_embeddings.iloc[:, 1:], bert_embeddings.iloc[:, 1:], metric='euclidean')
    # dot product
    pairwise_dot_prod = np.dot(bert_embeddings.iloc[:, 1:], bert_embeddings.iloc[:, 1:].T)
    # cosine similarity
    pairwise_cosine_similarities = np.dot(bert_embeddings.iloc[:, 1:], bert_embeddings.iloc[:, 1:].T)
    pairwise_cosine_similarities /= np.linalg.norm(bert_embeddings.iloc[:, 1:], axis=1, keepdims=True)
    pairwise_cosine_similarities /= np.linalg.norm(bert_embeddings.iloc[:, 1:], axis=1, keepdims=True).T

    mean_cs = pairwise_cosine_similarities.mean()
    print("Mean cosine similarity value: ", mean_cs)

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 8))
    sns.heatmap(pairwise_cosine_similarities, cmap='viridis', annot=False, vmin=0.3)
    plt.title(f"{args.embeddings_type} - cosine similarities")
    plt.xlabel("Species")
    plt.ylabel("Species")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    # plt.savefig(f"./{args.embeddings_type}_cosineSimilarities.pdf")
    plt.show()