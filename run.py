#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
import time
import argparse

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from sklearn import metrics
from model import Hypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils


def parse_args():
    parser = argparse.ArgumentParser(description='run.py for training, evaluating model')

    parser.add_argument('-mode', default='test', type=str,
                        help="choose mode from train or test")

    parser.add_argument('-cuda', default=1, type=int,
                        help="Use cuda for training and evaluating model, negative int will use cpu instead")

    parser.add_argument('-train_src', default='./data/AAPD/text_train', type=str,
                        help="File of training target sentences")
    parser.add_argument('-train_tgt', default='./data/AAPD/label_train', type=str,
                        help="File of training target sentences (label)")
    parser.add_argument('-dev_src', default='./data/AAPD/text_val', type=str,
                        help="File of validating target sentences")
    parser.add_argument('-dev_tgt', default='./data/AAPD/label_val', type=str,
                        help="File of validating target sentences (label)")
    parser.add_argument('-test_src', default='./data/AAPD/text_test', type=str,
                        help="File of validating target sentences")
    parser.add_argument('-test_tgt', default='./data/AAPD/label_test', type=str,
                        help="File of validating target sentences (label)")

    parser.add_argument('-vocab', default='./data/AAPD/vocab.json', type=str,
                        help="File of vocabulary")
    parser.add_argument('-output_file', default='./outputs/AAPD/test_outputs.txt', type=str,
                        help="File of vocabulary")

    parser.add_argument('-seed', default=0, type=int,
                        help="seed, default 0")
    parser.add_argument('-batch_size', default=32, type=int,
                        help="batch_size, default 32")
    parser.add_argument('-embed_size', default=256, type=int,
                        help="embed_size, default 256")
    parser.add_argument('-hidden_size', default=256, type=int,
                        help="hidden_size, default 256")
    parser.add_argument('-clip_grad', default=5.0, type=float,
                        help="gradient clipping, default: 5.0")
    parser.add_argument('-log_every', default=10, type=int,
                        help="log_every, default 10")
    parser.add_argument('-max_epoch', default=30, type=int,
                        help="max_epoch, default 30")
    parser.add_argument('-patience', default=5, type=int,
                        help="wait for how many iterations to decay learning rate default: 5")
    parser.add_argument('-max_num_trial', default=5, type=int,
                        help="terminate training after how many trials, default 5")
    parser.add_argument('-lr_decay', default=0.5, type=float,
                        help="lr_decay, default 0.5")
    parser.add_argument('-beam_size', default=5, type=int,
                        help="beam_size, default 5")
    parser.add_argument('-lr', default=0.001, type=float,
                        help="learning rate, default 0.001")
    parser.add_argument('-uniform_init', default=0.1, type=float,
                        help="uniformly initialize all parameters, default 0.1")
    parser.add_argument('-save_to', default='./checkpoints/AAPD/model.bin', type=str,
                        help="model save path, default ./checkpoints/model.bin")
    parser.add_argument('-valid_niter', default=2000, type=int,
                        help="perform validation after how many iterations, default: 2000")
    parser.add_argument('-dropout', default=0.3, type=float,
                        help="dropout, default 0.3")
    parser.add_argument('-max_decoding_time_step', default=70, type=int,
                        help="max_decoding_time_step, default 70")

    arguments = parser.parse_args()

    return arguments


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model: (NMT), NMT Model
    @param dev_data: (list of (src_sent, tgt_sent)), list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplexity on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences (or labels), compute corpus-level BLEU score.
    @param references: (List[List[str]]), a list of gold-standard reference target sentences (or labels)
    @param hypotheses: (List[Hypothesis]), a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def compute_hamming_loss(references: List[List[str]],
                         hypotheses: List[Hypothesis],
                         tgt_dictionary: VocabEntry) -> Dict[str, float]:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references: (List[List[str]]), a list of gold-standard reference target sentences (or labels)
    @param hypotheses: (List[Hypothesis]), a list of hypotheses, one for each reference
    @param tgt_dictionary: (Dict[str, int]), a dictionary of tgt sentences (or labels)
    @returns hamming_loss: hamming loss
    """

    def sentence_ids_to_multi_ones_hot_vector(y: List[str],
                                              dictionary: VocabEntry) -> np.array:
        total_length = len(dictionary)
        ones_hot = np.zeros(total_length, dtype=np.int)
        hot_indices = dictionary.words2indices(y)
        ones_hot[hot_indices] = 1
        # ignore the following words '<pad>' '<s>' '</s>' '<unk>'
        return ones_hot[4:]

    def sentences_ids_to_multi_ones_hot_vectors(ys: List[List[str]],
                                                dictionary: VocabEntry) -> np.array:
        return np.array([sentence_ids_to_multi_ones_hot_vector(y, dictionary) for y in ys],
                        dtype=np.int)

    references_ones_hot_vectors = sentences_ids_to_multi_ones_hot_vectors(references, tgt_dictionary)
    hypotheses_ones_hot_vectors = sentences_ids_to_multi_ones_hot_vectors([hypothese.value for hypothese in hypotheses],
                                                                          tgt_dictionary)
    hamming_loss = metrics.hamming_loss(references_ones_hot_vectors, hypotheses_ones_hot_vectors)
    macro_f1 = metrics.f1_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='macro')
    macro_precision = metrics.precision_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='macro')
    macro_recall = metrics.recall_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='macro')

    micro_f1 = metrics.f1_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='micro')
    micro_precision = metrics.precision_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='micro')
    micro_recall = metrics.recall_score(references_ones_hot_vectors, hypotheses_ones_hot_vectors, average='micro')

    results = dict(hamming_loss=hamming_loss,
                   macro_f1=macro_f1,
                   macro_precision=macro_precision,
                   macro_recall=macro_recall,
                   micro_f1=micro_f1,
                   micro_precision=micro_precision,
                   micro_recall=micro_recall, )

    return results


def train(args):
    """ Train the NMT Model.
    @param args, args from cmd line
    """
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')

    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = args.batch_size
    clip_grad = args.clip_grad
    valid_niter = args.valid_niter
    log_every = args.log_every
    model_save_path = args.save_to

    vocab = Vocab.load(args.vocab)

    model = NMT(embed_size=args.embed_size,
                hidden_size=args.hidden_size,
                dropout_rate=args.dropout,
                vocab=vocab)
    model.train()

    uniform_init = args.uniform_init
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    # seem to be useless
    # vocab_mask = torch.ones(len(vocab.tgt))
    # vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:%d" % args.cuda if args.cuda >= 0 else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents)  # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # TODO findout grad_norm useless or not
            # grad_norm seem to be useless
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch: [%d] | iter: [%d] | avg. loss: [%.2f] | avg. ppl: [%.2f] |'
                      'cum. examples: [%d] | speed: [%.2f words/sec] | time elapsed: [%.2f sec]' % (
                          epoch, train_iter,
                          report_loss / report_examples,
                          math.exp(
                              report_loss / report_tgt_words),
                          cum_examples,
                          report_tgt_words / (
                                  time.time() - train_time),
                          time.time() - begin_time
                      ),
                      file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch: [%d] | iter: [%d] | cum. loss: [%.2f] | cum. ppl: [%.2f] | cum. examples: [%d]' % (
                    epoch, train_iter,
                    cum_loss / cum_examples,
                    np.exp(
                        cum_loss / cum_tgt_words),
                    cum_examples
                ),
                      file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)  # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('*** validation *** iter: [%d] | dev. ppl: [%f]' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < args.patience:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args.patience):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == args.max_num_trial:
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == args.max_epoch:
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def decode(args):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args: args from cmd line
    """

    print("load test source sentences from [{}]".format(args.test_src), file=sys.stderr)
    test_data_src = read_corpus(args.test_src, source='src')
    if args.test_tgt:
        print("load test target sentences from [{}]".format(args.test_tgt), file=sys.stderr)
        test_data_tgt = read_corpus(args.test_tgt, source='tgt')

    print("load model from {}".format(args.save_to), file=sys.stderr)
    model = NMT.load(args.save_to)

    if args.cuda >= 0:
        model = model.to(torch.device("cuda:%d" % args.cuda))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=args.beam_size,
                             max_decoding_time_step=args.max_decoding_time_step)

    if args.test_tgt:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        results = compute_hamming_loss(test_data_tgt, top_hypotheses, model.vocab.tgt)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)
        for result, val in results.items():
            print('%s: %.4f' % (result, val), file=sys.stderr)

    with open(args.output_file, 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')
    pass


def beam_search(model: NMT,
                test_data_src: List[List[str]],
                beam_size: int,
                max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model: (NMT), NMT Model
    @param test_data_src: (List[List[str]]), List of sentences (words) in source language, from test set.
    @param beam_size: (int), beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step: (int), maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size,
                                             max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training:
        model.train(was_training)

    return hypotheses


def main():
    """ Main func.
    """
    args = parse_args()

    # Check pytorch version
    assert torch.__version__ >= "1.0.0", "Expect pytorch version higher than 1.0.0, but got".format(torch.__version__)

    # seed the random number generators
    seed = args.seed
    torch.manual_seed(seed)
    if args.cuda >= 0:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()