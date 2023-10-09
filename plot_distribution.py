import os
import sys
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np

features_path = "/nas/staff/data_work/Andreas/HearTheSpecies/bird-recognition/bird-description-fusion/ast-embeddings/features.csv"
features_df = pd.read_csv(features_path)
features_df = features_df.rename(columns={'file': 'filename'})
features_df['filename'] = features_df['filename'].apply(lambda x: x.lstrip('/'))
features_df['species'] = features_df['filename'].apply(lambda x: x.split('/')[0])
features_df


with open("/nas/staff/data_work/AG/BIRDS/baseline_embeddings/mapping.json", "r") as f:
    mapping = json.load(f)

features_df["species"] = features_df["species"].apply(lambda x: x.lower().replace("_", " "))
features_df["species"] = [mapping[s] if s in mapping.keys() else s for s in features_df["species"]]

value_counts = features_df['species'].value_counts()

# Plotting the value counts using a bar plot
plt.bar(value_counts.index, value_counts.values, width=1)
plt.xlabel('Species')
plt.ylabel('Samples')
plt.xticks(np.arange(0, len(value_counts), 5), np.arange(0, len(value_counts), 5))
plt.title('Number of Samples')
plt.tight_layout()
# plt.savefig('./species_distribution.pdf')
plt.show()