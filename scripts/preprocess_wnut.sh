#!/usr/bin/env bash
#Download WNUT 2017
mkdir -p ../data/wnut17
git clone https://github.com/leondz/emerging_entities_17.git ../data/wnut17/source

# Load and reformat OntoNotes
cd ../code
python3 -m data.preprocess_wnut