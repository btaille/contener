import logging
import os
import shutil

import torch
from torch import nn


def save_checkpoint(state, is_best, checkpoint='checkpoint.pth.tar', best='model_best.pth.tar'):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)


def load_checkpoint(filename):
    if os.path.isfile(filename):
        logging.info("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=lambda storage, location: storage)
        epoch = checkpoint['epoch']
        logging.info("=> loaded checkpoint @ epoch {}".format(epoch))
    else:
        logging.info("=> no checkpoint found at '{}'".format(filename))
        return None

    return checkpoint


def mask(seqlens):
    mask = torch.zeros(len(seqlens), max(seqlens), dtype=torch.int)
    for i, l in enumerate(seqlens):
        mask[i, :l] = torch.ones((l))
    return mask


def weight_init(model):
    for m in model.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                print(m, type(m), name, param.requires_grad)
                if param.requires_grad:
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0.001)

        elif type(m) in [nn.Linear]:
            for name, param in m.named_parameters():
                print(m, type(m), name, param.requires_grad)
                if param.requires_grad:
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0.001)
