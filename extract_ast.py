import argparse
import audiofile
import glob
import os
import tqdm
import torch
import numpy as np
import pandas as pd

from transformers import ASTFeatureExtractor
from transformers import AutoModelForAudioClassification

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AST Embeddings")
    parser.add_argument("--data", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.data, "**", "*.mp3"))
    
    os.makedirs(os.path.join(os.path.dirname(args.dest), "ast-embeddings"), exist_ok=True)
    embs = []
    feature_extractor = ASTFeatureExtractor()
    model = AutoModelForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        output_hidden_states=True
    ).to("cuda:0")
    emb_filenames = []
    for index, file in tqdm.tqdm(enumerate(files), total=len(files), desc="AST"):
        filename = f"{index:010}.npy"
        waveform = audiofile.read(file, always_2d=True)[0]
        waveform = waveform.mean(0, keepdims=True)
        inputs = feature_extractor(
            waveform, 
            sampling_rate=16000, 
            padding="max_length", 
            return_tensors="pt"
        )
        input_values = inputs.input_values
        with torch.no_grad():
            outputs = model(input_values.to("cuda:0"))
            outputs = outputs[1][-1].cpu().numpy()
            np.save(
                os.path.join(os.path.dirname(args.dest), "ast-embeddings", filename),
                outputs
            )
            emb_filenames.append(filename)
            embs.append(outputs.mean(1))
        
    embs = np.concatenate(embs)
    print(embs.shape)
    files = [file.replace(args.data, "") for file in files]
    df = pd.DataFrame(
        data=embs,
        columns=[f"Neuron_{x}" for x in range(embs.shape[1])],
        index=pd.Index(files, name="file")
    ).reset_index()
    print(df.info())
    df.to_csv(args.dest, index=False)

    pd.DataFrame(
        data=emb_filenames,
        columns=["features"],
        index=pd.Index(files, name="file")
    ).reset_index().to(os.path.join(os.path.dirname(args.dest), "ast-embeddings", "features.csv"), index=False)
