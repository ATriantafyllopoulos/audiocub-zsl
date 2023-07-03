import argparse
import glob
import os
import tqdm
import librosa
import numpy as np
import pandas as pd
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CNN14 Embeddings")
    parser.add_argument("--data", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.data, "**", "*.mp3"))
    
    embs = []
    at = AudioTagging(checkpoint_path=None, device='cuda:0')
    for file in tqdm.tqdm(files, total=len(files), desc="CNN14"):
        try:
            (audio, _) = librosa.core.load(file, sr=32000, mono=True)
            audio = audio[None, :]
            (clipwise_output, embedding) = at.inference(audio)
            embs.append(embedding)
        except:
            embs.append(np.zeros((1, 2048)))
        
    embs = np.concatenate(embs)
    print(embs.shape)
    files = [file.replace(args.data, "") for file in files]
    df = pd.DataFrame(
        data=embs,
        columns=[f"Neuron_{x}" for x in range(embs.shape[1])],
        index=pd.Index(files, name="file")
    ).reset_index()
    print(df.info())
    os.makedirs(os.path.dirname(args.dest), exist_ok=True)
    df.to_csv(args.dest, index=False)
