# Zero-shot bird classification

Map acoustic vectors to BERT embeddings.
Embeddings extracted from Birds of Europe Princeton Field Guide.

## Installation

For running the code, simply do: `pip install -r requirements.txt`

If you additionally need to extract embeddings, also do: `pip install -r requirements.embeddings.txt`

## Data

The raw audio data can be found in "/data/eihw-gpu6/trianand/BIRDS/audio".
The text and audio embeddings can be found in "/nas/staff/data_work/Andreas/HearTheSpecies/bird-recognition/bird-description-fusion/".

## Running the code:
A sample bash call for running the code is:

```bash
python main.py \
    meta.results_root="foo" \
    meta.audio_features="/nas/staff/data_work/Andreas/HearTheSpecies/bird-recognition/bird-description-fusion/features.teresa.csv" \
    meta.text_features="/nas/staff/data_work/Andreas/HearTheSpecies/bird-recognition/bird-description-fusion/bert_embeddings.csv"
```

This is using some baseline audio features extracted by Teresa.