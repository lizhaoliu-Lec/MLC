#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py
"""

import math
import random
import numpy as np


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents: (list[list[str]]), list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token: (str), padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    # YOUR CODE HERE (~6 Lines)

    # (1) get max length
    maxLen = -1
    for sent in sents:
        if len(sent) > maxLen:
            maxLen = len(sent)

    # (2) pad every sent in sents
    for sent in sents:
        pad_length = maxLen - len(sent)
        # pad shorter sent with pad_length * pad_token
        sent = sent + pad_length * [pad_token]
        sents_padded.append(sent)

    # END YOUR CODE

    return sents_padded


def read_corpus(file_path, source, sep=' ', border='\n'):
    """ Read file, where each sentence is surrounded by a border (default '\n') and split by end (default ' ').
    @param file_path: (str), path to file containing corpus
    @param source: (str), "tgt" or "src" indicating whether text
        is of the source language or target language
    @param sep: (str), sep token that separate each word, default ' '
    @param border: (str), border token that surround each sentence, default '\n'
    """
    data = []
    for line in open(file_path):
        sent = line.strip(border).split(sep)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data,
               batch_size,
               shuffle=False,
               shuffle_target=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data: (list of (src_sent, tgt_sent)), list of tuples containing source and target sentence
    @param batch_size: (int), batch size
    @param shuffle: (boolean), whether to randomly shuffle the dataset
    @param shuffle_target: (boolean), whether to randomly shuffle the dataset's label
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)

        src_sents = [e[0] for e in examples]
        if shuffle_target:
            tgt_sents = []
            for e in examples:
                tgt = e[1]
                # chop the start and end token off
                start, end = tgt[0], tgt[-1]
                real_tgt = tgt[1:-1]
                if len(tgt) > 3:
                    random.shuffle(real_tgt)
                    tgt_sents.append([start] + real_tgt + [end])
                else:
                    tgt_sents.append(tgt)

        else:
            tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
