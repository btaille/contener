#!/usr/bin/env bash
tar -zxvf ../data/ontonotes-release-5.0_LDC2013T19.tgz

### Download and untar CoNLL2012 scripts and annotations
curl -o ../data/conll-2012-train.tar.gz http://conll.cemantix.org/2012/download/conll-2012-train.v4.tar.gz
curl -o ../data/conll-2012-dev.tar.gz http://conll.cemantix.org/2012/download/conll-2012-development.v4.tar.gz
curl -o ../data/conll-2012-test.tar.gz http://conll.cemantix.org/2012/download/test/conll-2012-test-key.tar.gz
curl -o ../data/conll-2012-scripts.tar.gz http://conll.cemantix.org/2012/download/conll-2012-scripts.v3.tar.gz

tar -zxvf ../data/conll-2012-train.tar.gz
tar -zxvf ../data/conll-2012-dev.tar.gz
tar -zxvf ../data/conll-2012-test.tar.gz
tar -zxvf ../data/conll-2012-scripts.tar.gz

# Convert *_skel to *_conll
cd ../data
bash conll-2012/v3/scripts/skeleton2conll.sh -D ontonotes-release-5.0/data/files/data conll-2012

# Load and reformat OntoNotes
cd ../code
python3 -m data.preprocess_ontonotes