import logging

import numpy as np
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm

from data.data_iterator import ids2seq
from utils.evaluation import compute_score, compute_score_overlap, add_score, add_score_overlap
from utils.torch_utils import save_checkpoint, weight_init

class Model(nn.Module):
    def train_step(self, batch, optimizer, grad_clipping=0, optimizer_step=True, gradient_accumulation=1,
                   **kwargs):
        self.forward(batch, **kwargs)
        loss = self.loss(batch) / gradient_accumulation
        loss.backward()

        if grad_clipping > 0:
            clip_grad_value_(self.parameters(), grad_clipping)

        if optimizer_step:
            optimizer.step()
            self.zero_grad()

    def run_epoch(self, iterators, epoch, optimizer, writer, grad_clipping=0, train_key="train",
                  gradient_accumulation=1, **kwargs):
        self.train()
        iterators[train_key].reinit()

        losses = []

        n_batches = iterators[train_key].nbatches

        for i in tqdm(range(n_batches)):
            n_iter = epoch * n_batches + i

            batch = iterators[train_key].__next__()

            self.train_step(batch, optimizer, grad_clipping=grad_clipping,
                            optimizer_step=(i + 1) % gradient_accumulation == 0,
                            gradient_accumulation=gradient_accumulation, **kwargs)

            writer.add_scalars("ner_loss", {"train": batch["ner_loss"].item()}, n_iter)
            losses.append(batch["ner_loss"].item())

            if "loss" in batch.keys():
                writer.add_scalars("loss", {"train": batch["loss"].item()}, n_iter)

        return losses

    def train_loop(self, iterators, optimizer, run_dir,
                   epochs=100, min_epochs=0, patience=5, epoch_start=0,
                   best_f1=None, epochs_no_improv=None,
                   grad_clipping=0,
                   overlap=None, train_entities=None, train_key="train", dev_key="dev",
                   eval_on_train=False, gradient_accumulation=1, **kwargs):

        logging.info("Starting train loop: {} epochs; {} min; {} patience".format(epochs, min_epochs, patience))

        if best_f1 is None:
            best_f1 = 0

        if epochs_no_improv is None:
            epochs_no_improv = 0

        if not train_key == "train":
            patience = 0

        if patience and epoch_start > min_epochs and epochs_no_improv >= patience:
            logging.info("Early stopping after {} epochs without improvement.".format(patience))

        else:
            writer = SummaryWriter(run_dir)
            for epoch in range(epoch_start, epochs):
                logging.info("Epoch {}/{} :".format(epoch + 1, epochs))
                train_losses = self.run_epoch(iterators, epoch, optimizer, writer,
                                              grad_clipping=grad_clipping, train_key=train_key,
                                              gradient_accumulation=gradient_accumulation)
                n_iter = (epoch + 1) * len(list(train_losses))


                if eval_on_train:
                    logging.info("Train eval")
                    self.evaluate(iterators["ner"][train_key])

                _, ner_loss, ner_scores = self.evaluate(iterators[dev_key], overlap=overlap,
                                                            train_entities=train_entities)

                logging.info("Train NER Loss : {}".format(np.mean(train_losses)))
                logging.info("Dev NER Loss : {}".format(ner_loss))

                if overlap is None:
                    if "ner" in iterators.keys():
                        add_score(writer, ner_scores, n_iter)
                else:
                    if "ner" in iterators.keys():
                        add_score_overlap(writer, ner_scores, n_iter, task="ner")

                f1 = ner_scores["ALL"]["f1"]

                if f1 > best_f1:
                    logging.info(f"New best NER F1 score on dev : {f1}")
                    logging.info("Saving model...")
                    best_f1 = f1
                    epochs_no_improv = 0
                    is_best = True

                else:
                    epochs_no_improv += 1
                    is_best = False

                state = {'epoch': epoch + 1,
                         'epochs_no_improv': epochs_no_improv,
                         'model': self.state_dict(),
                         'scores': ner_scores,
                         'optimizer': optimizer.state_dict()
                         }
                save_checkpoint(state, is_best, checkpoint=run_dir + 'ner_checkpoint.pth.tar',
                                best=run_dir + 'ner_best.pth.tar')

                writer.add_scalars("ner_loss", {"dev": ner_loss}, n_iter)

                if patience and epoch > min_epochs and epochs_no_improv >= patience:
                    logging.info(
                        f"Early stopping after {patience} epochs without improvement on NER.")
                    break

            writer.export_scalars_to_json(run_dir + "all_scalars.json")
            writer.close()

    def evaluate(self, iterator, overlap=None, train_entities=None,
                     correct_iob=False):
        self.eval()
        iterator.reinit()
        preds = []
        losses = []
        labels = []

        entities = sorted([k.split("-")[1] for k in self.decoder.tag2idx.keys() if k[0] == "B"])

        if overlap is not None:
            assert train_entities is not None
            sents = []

        for data in tqdm(iterator):
            self.forward(data)
            predictions = data["ner_output"]
            predictions = self.decoder.decode(predictions, data["nwords"])
            loss = data["ner_loss"]
            losses.append(loss.item())

            if overlap is not None:
                sents.extend(data["sents"])

            for i in range(len(data["tags"])):
                labels.append(
                    ids2seq(data["tags"][i, :data["nwords"][i]].tolist(), self.decoder.idx2tag))
                preds.append(predictions[i])

        if overlap is None:
            return preds, np.mean(losses), compute_score(preds, labels, entities=entities, scheme=self.scheme,
                                                         correct=correct_iob)
        else:
            return preds, np.mean(losses), compute_score_overlap(sents, preds, labels, overlap, train_entities,
                                                                 scheme=self.scheme, correct=correct_iob)

    def predict(self, iterator):
        self.eval()
        iterator.reinit()
        preds = []

        for data in tqdm(iterator):
            predictions = self.forward(data)
            preds.append(self.decoder.decode(predictions, data["nwords"]))

        return preds


class EmbedderEncoderDecoder(Model):
    def __init__(self, embedder, encoder, decoder, scheme="iobes"):
        super(EmbedderEncoderDecoder, self).__init__()
        # parameters
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder

        self.scheme = scheme

        if self.encoder is not None:
            weight_init(self.encoder)
        weight_init(self.decoder)

    def forward(self, data):
        data.update({"embeddings": self.embedder(data)["embeddings"]})

        if self.encoder is not None:
            data.update({"encoded": self.encoder(data)["output"]})
        else:
            data.update({"encoded": data["embeddings"]})

        self.decoder(data)
        if self.decoder.supervision in data.keys():
            data.update({"ner_loss": self.decoder.loss(data)})

        return data

    def loss(self, data):
        return self.decoder.loss(data)
