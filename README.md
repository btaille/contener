ContENER
====

["Contextualized Embeddings in Named-Entity Recognition: An Empirical Study on Generalization", ECIR 2020](https://arxiv.org/pdf/2001.08053.pdf)  
This code enables to compare the impact of BERT, ELMo, Flair and GloVe representations on NER generalization both to unseen mentions and out-of-domain.  

Test entities are separated in 3 categories:
- **Exact Match** if they are in the training set in the exact same case sensitive form and tagged with the same type.
- **Partial Match** if at least one of their non stop words is in an entity of same type in the training set.
- **New** otherwise, i.e. if all their non stop words are unseen for this type.

### Requirements
The code is written in Python 3.6 with the following main dependencies:

* Pytorch 1.3.1
* AllenNLP 0.8.2
* pytorch-pretrained-bert 0.6.1
* flair 0.4.1

### Docker 
A Dockerfile with the corresponding GPU environment and jupyterlab is provided in `docker/`
```bash
cd docker
nvidia-docker build -t contener -f Dockerfile_contener.gpu .
nvidia-docker run -it -p 8888:8888 -v <absolute_path>/:notebooks/ contener
``` 

## Data Setup and Preprocessing
We experimented with the English versions of CoNLL03, [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) and [WNUT 2017](https://github.com/leondz/emerging_entities_17) datasets.  

##### 1) GloVe and ELMo embeddings
Run `scripts/download_glove_and_elmo.sh` to download GLoVe embeddings and ELMo 5.5B weights.
##### 2) CoNLL03
You will need to find the `eng.train`, `eng.testa` and `eng.testb`  files on your own.  
Then place them in `data/conll03/source/` and run `scripts/preprocess_conll03.sh`
##### 3) WNUT 17
Simply run `scripts/preprocess_wnut.sh` to download and preprocess WNUT17.
##### 4) OntoNotes 5.0
Download `ontonotes-release-5.0_LDC2013T19.tgz` from the [LDC](https://catalog.ldc.upenn.edu/LDC2013T19) and place it in `data/`.  
Then run `scripts/preprocess_ontonotes.sh`

## Training
The results presented in the paper were obtained using the following configurations:
 - embedder in `glove`, `glove char`, `elmo`, `elmo_zero`, `flair`, `bert-large-fb` (feature-based).
 - encoder in `bilstm`, `map`
 - decoder in `crf` 

when running:
```bash
python train_ner.py -ds $dataset -emb $embedder -enc $encoder -dec $decoder -d 0.5 -bs 64 -ep 100 -p 5
```

Additionally:
 - embedder can be `bert-large` for the finetuned version of BERT, `bert-base` or `bert-base-fb`, or any space separated list of embeddings that will be concatenated as in `glove char`.
 - decoder can be `softmax` 

## Out-of-domain Evaluation

For out-of-domain evaluation, once training is complete for a given configuration, run:

```bash
python test_ood.py -ds $dataset -ood_ds $ood_dataset -emb $embedder -enc $encoder -dec $decoder
```


## Reference
If you find any of this work useful, please cite our paper as follows:
```
@misc{taille2020contextualized,
    title={Contextualized Embeddings in Named-Entity Recognition: An Empirical Study on Generalization},
    author={Bruno Taillé and Vincent Guigue and Patrick Gallinari},
    year={2020},
    eprint={2001.08053},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


