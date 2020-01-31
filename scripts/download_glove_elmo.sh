#!/bin/bash
# Download GloVe
glove_path='http://nlp.stanford.edu/data/glove.840B.300d.zip'
mkdir -p ../embeddings/glove.840B
curl -LO $glove_path
unzip glove.840B.300d.zip -d ../embeddings/glove.840B/
rm glove.840B.300d.zip

# Download ELMo weights
elmo_options_path='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
elmo_weights_path='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
mkdir -p ../embeddings/elmo
cd ../embeddings/elmo
curl -LO $elmo_options_path
curl -LO $elmo_weights_path
