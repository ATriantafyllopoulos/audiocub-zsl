# Zero-shot bird classification

Map acoustic vectors to class (e.g. BERT) embeddings.
Embeddings extracted from Birds of Europe Princeton Field Guide.

## Installation

For running the code, simply do: `pip install -r requirements.txt`

If you additionally need to extract embeddings, also do: `pip install -r requirements.embeddings.txt`

## Data

#### Old
The raw audio data can be found in "/data/eihw-gpu6/trianand/BIRDS/audio".
The text and audio embeddings can be found in "/nas/staff/data_work/Andreas/HearTheSpecies/bird-recognition/bird-description-fusion/".

#### New
The text embeddings, functional features, and their concatenations can be found in "/embeddings/baseline_embeddings/".

## Running the code:
A sample bash call for running the code is:

```bash
python main.py \
    meta.results_root="foo" \
    meta.audio_features="/nas/staff/data_work/Andreas/HearTheSpecies/bird-recognition/bird-description-fusion/ast.csv" \
    meta.text_features="./embeddings/baseline_embeddings/bert_embeddings.csv" \
    meta.mapping_path="./embeddings/baseline_embeddings/mapping.json" \
    meta.predefined_clf_splits="./splits/bird_classification/"
    meta.predefined_zsl_splits="./splits/zsl-5fold/"
```

This is using AST audio features and BERT text embeddings.

## Running the additional scripts:
In the embeddings folder, the BLH data is sometimes also called history or dryad due to previous  iterations.

### Obtain processed baseline embeddings from the functional features
A sample bash call for running the code is:

```bash
python baseline_embeddings.py \
    --audiodir "foo" \
    --storepath "foo" \
    --meta-information "./meta_information/avonet.xlsx" \
    --mapping-path "./embeddings/baseline_embeddings/mapping.json" \
    --meta-type "avonet"
```

### Concatenate embeddings
```bash
python concatenate_embeddings.py \
    --bert "./embeddings/baseline_embeddings/bert_embeddings.csv" \
    --avonet "./embeddings/baseline_embeddings/avonet_embeddings.csv" \
    --blh "./embeddings/baseline_embeddings/history_embeddings.csv" \
    --storepath "foo" \
```

### Show cosine similarity for (S)BERT embeddings
```bash
python similarity_bert.py \
    --bert-embeddings "./embeddings/baseline_ambeddings/" \
    --embeddings_type "bert"
```

### Extract AST features
```bash
python extract_ast.py \
    --data "foo" \
    --dest "./results/"
```

### Extract CNN14 embeddings
```bash
python extract_cnn14_embeddings.py \
    --data "foo" \
    --dest "./results/"
```
