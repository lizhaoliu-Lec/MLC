#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model.py: NMT model
"""
from collections import namedtuple
import sys
from typing import List, Tuple
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from embeddings import Embeddings

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            if len(alpha) != 2:
                raise ValueError('expect alpha to be a list of 2 element, but got `%d`' % len(alpha))
            self.alpha = torch.tensor(alpha)
        else:
            raise ValueError('expect alpha to be a float, int or list, but got `%s`' % type(alpha))
        self.size_average = size_average

    def forward(self, predict: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:

        batch_size = predict.size(1)
        length = predict.size(0)

        # predict: (len, b, tgt_vocab_size)
        # target: (len, b)
        # mask: (len, b)

        # log_prob: (len, b, tgt_vocab_size) -> (len, b)
        log_prob = F.log_softmax(predict, dim=-1)
        log_prob = torch.gather(log_prob,
                                index=target.unsqueeze(-1),
                                dim=-1).squeeze(-1) * mask
        log_prob = log_prob.view(-1)
        prob = torch.exp(log_prob)

        if self.alpha is not None:
            if self.alpha.type() != predict.type():
                print('*** self.alpha.type() ***', self.alpha.type())
                print('*** predict.type() ***', predict.type())
                self.alpha = self.alpha.type_as(predict)
            # TODO, figure it out
            # not understand yet
            alpha = self.alpha.gather(0, target.view(-1))
            log_prob = log_prob * alpha

        loss = (1 - prob) ** self.gamma * log_prob

        loss = loss.view(length, batch_size)
        if self.size_average:
            return loss.mean(dim=0)
        else:
            return loss.sum(dim=0)


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidirectional LSTM Encoder
        - Unidirectional LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab,
                 dropout_rate=0.2,
                 loss_type='cross_entropy',
                 **kwargs):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = Embeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        allow_loss_type = ['cross_entropy', 'focal_loss']

        if loss_type not in allow_loss_type:
            raise ValueError('loss_type must be one of `%s`, but got `%s`' % (allow_loss_type, loss_type))

        self.loss_type = loss_type

        # init variable for focal loss
        if self.loss_type == 'focal_loss':
            self.gamma = kwargs.pop('gamma', 2)

        # self.encoder (Bidirectional LSTM with bias)
        # self.decoder (LSTM Cell with bias)
        # self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        # self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        # self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        # self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        # self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        # self.dropout (Dropout Layer)
        ###
        # Use the following docs to properly initialize these variables:
        # LSTM:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        # LSTM Cell:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        # Linear Layer:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        # Dropout Layer:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size,
                               bidirectional=True, bias=True)
        self.decoder = nn.LSTMCell(input_size=embed_size + hidden_size,
                                   hidden_size=hidden_size,
                                   bias=True)

        self.h_projection = nn.Linear(in_features=2 * hidden_size,
                                      out_features=hidden_size,
                                      bias=False)
        self.c_projection = nn.Linear(in_features=2 * hidden_size,
                                      out_features=hidden_size,
                                      bias=False)

        self.att_projection = nn.Linear(in_features=2 * hidden_size,
                                        out_features=hidden_size,
                                        bias=False)
        self.combined_output_projection = nn.Linear(in_features=3 * hidden_size,
                                                    out_features=hidden_size,
                                                    bias=False)
        self.target_vocab_projection = nn.Linear(in_features=hidden_size,
                                                 out_features=len(vocab.tgt),
                                                 bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    # def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
    #     """ Take a mini-batch of source and target sentences, compute the log-likelihood of
    #     target sentences under the language models learned by the NMT system.
    #
    #     @param source: (List[List[str]]), list of source sentence tokens
    #     @param target: (List[List[str]]), list of target sentence tokens, wrapped by `<s>` and `</s>`
    #
    #     @returns scores: (Tensor), a variable/tensor of shape (b, ) representing the
    #                                 log-likelihood of generating the gold-standard target sentence for
    #                                 each example in the input batch. Here b = batch size.
    #     """
    #     # Compute sentence lengths
    #     source_lengths = [len(s) for s in source]
    #
    #     # Convert list of lists into tensors
    #     source_padded = self.vocab.src.to_input_tensor(
    #         source, device=self.device)  # Tensor: (src_len, b)
    #     target_padded = self.vocab.tgt.to_input_tensor(
    #         target, device=self.device)  # Tensor: (tgt_len, b)
    #
    #     # Run the network forward:
    #     # 1. Apply the encoder to `source_padded` by calling `self.encode()`
    #     # 2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
    #     # 3. Apply the decoder to compute combined-output by calling `self.decode()`
    #     # 4. Compute log probability distribution over the target vocabulary using the
    #     # combined_outputs returned by the `self.decode()` function.
    #
    #     # 1. Apply the encoder to `source_padded` by calling `self.encode()`
    #     enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
    #     # 2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
    #     enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
    #     # 3. Apply the decoder to compute combined-output by calling `self.decode()`
    #     combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
    #     # 4. Compute log probability distribution over the target vocabulary using the
    #     # combined_outputs: (tgt_len - 1, b, h) -> target_vocab_projection: (tgt_len - 1, b, tgt_vocab_size)
    #     # -> P: (tgt_len - 1, b, tgt_vocab_size)
    #     target_vocab_projection = self.target_vocab_projection(combined_outputs)
    #     P = F.log_softmax(target_vocab_projection, dim=-1)
    #
    #     # Zero out, probabilities for which we have nothing in the target text
    #     # target_padded: (tgt_len, b) -> target_masks: (tgt_len, b)
    #     target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
    #
    #     # Compute log probability of generating true target words
    #     # P: (tgt_len - 1, b, tgt_vocab_size), index: (tgt_len - 1, b), target_masks: (tgt_len - 1, b)
    #     # -> target_gold_words_log_prob: (tgt_len - 1, b)
    #     target_gold_words_log_prob = torch.gather(P,
    #                                               index=target_padded[1:].unsqueeze(-1),
    #                                               dim=-1).squeeze(-1) * target_masks[1:]
    #     #
    #     # scores: (b,)
    #     scores = target_gold_words_log_prob.sum(dim=0)
    #     return scores
    #
    #     # target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
    #     #
    #     # self.focal_loss.to(self.device)
    #     # scores = self.focal_loss(target_vocab_projection, target_padded[1:], target_masks[1:])
    #     # return scores

    def forward(self, source: List[List[str]], target: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source: (List[List[str]]), list of source sentence tokens
        @param target: (List[List[str]]), list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores: (Tensor), a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(
            source, device=self.device)  # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(
            target, device=self.device)  # Tensor: (tgt_len, b)

        # Run the network forward:
        # 1. Apply the encoder to `source_padded` by calling `self.encode()`
        # 2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        # 3. Apply the decoder to compute combined-output by calling `self.decode()`
        # 4. Compute log probability distribution over the target vocabulary using the
        # combined_outputs returned by the `self.decode()` function.

        # 1. Apply the encoder to `source_padded` by calling `self.encode()`
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        # 2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        # 3. Apply the decoder to compute combined-output by calling `self.decode()`
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        # 4. Compute log probability distribution over the target vocabulary using the
        # combined_outputs: (tgt_len - 1, b, h) -> target_vocab_projection: (tgt_len - 1, b, tgt_vocab_size)
        # -> P: (tgt_len - 1, b, tgt_vocab_size)
        target_vocab_projection = self.target_vocab_projection(combined_outputs)

        # logits: (tgt_len - 1, b, tgt_vocab_size)
        logits = F.softmax(target_vocab_projection, dim=-1)

        return logits, target_padded

    def compute_loss(self,
                     logits: torch.Tensor,
                     target_padded: torch.Tensor):
        # target_logits: (tgt_len - 1, b, tgt_vocab_size)
        # target_padded: (tgt_len, b)

        # target_masks: (tgt_len, b)
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
        # P: (tgt_len - 1, b, tgt_vocab_size)
        P = torch.log(logits)

        if self.loss_type == 'cross_entropy':
            # loss: (tgt_len - 1, b)
            loss = -1 * torch.gather(P,
                                     index=target_padded[1:].unsqueeze(-1),
                                     dim=-1).squeeze(-1) * target_masks[1:]
        elif self.loss_type == 'focal_loss':
            # loss: (tgt_len - 1, b, tgt_vocab_size)
            loss = -1 * (1 - logits) ** self.gamma * P
            # loss: (tgt_len - 1, b)
            loss = torch.gather(loss,
                                index=target_padded[1:].unsqueeze(-1),
                                dim=-1).squeeze(-1) * target_masks[1:]

        batch_size = logits.size(1)
        loss = torch.sum(loss) / batch_size

        return loss

    def compute_micro_f1(self,
                         logits: torch.Tensor,
                         target_padded: torch.Tensor) -> float:

        def sentence_ids_to_multi_ones_hot_vector(y: List[int]) -> np.array:
            total_length = len(self.vocab.tgt)
            ones_hot = np.zeros(total_length, dtype=np.int)
            hot_indices = y
            ones_hot[hot_indices] = 1
            return ones_hot

        def sentences_ids_to_multi_ones_hot_vectors(ys: List[List[int]]) -> np.array:
            return np.array([sentence_ids_to_multi_ones_hot_vector(y) for y in ys],
                            dtype=np.int)

        # logits: (tgt_len - 1, b, tgt_vocab_size)
        # target_padded: (tgt_len, b)

        # target_padded: (tgt_len, b) -> (tgt_len - 1, b)
        target_padded = target_padded[1:]

        # pred: (tgt_len - 1, b)
        pred = torch.argmax(logits, dim=2)

        # convert to list
        # target_padded: (tgt_len - 1, b)
        # (tgt_len - 1, b)
        target_padded = target_padded.tolist()
        pred = pred.tolist()

        target_ones_hot_vectors = sentences_ids_to_multi_ones_hot_vectors(target_padded)
        pred_ones_hot_vectors = sentences_ids_to_multi_ones_hot_vectors(pred)

        micro_f1 = metrics.f1_score(target_ones_hot_vectors, pred_ones_hot_vectors, average='micro')

        return micro_f1

    def encode(self,
               source_padded: torch.Tensor,
               source_lengths: List[int]) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded: (Tensor), Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths: (List[int]), List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens: (Tensor), Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state: (tuple(Tensor, Tensor)), Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        # YOUR CODE HERE (~ 8 Lines)
        # TODO:
        # 1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        # src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        # that there is no initial hidden state or cell for the decoder.
        #
        # 2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        # - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        # - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        # - Note that the shape of the tensor returned by the encoder is (src_len, b, h*2) and we want to
        # return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        #
        # 3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        # - `init_decoder_hidden`:
        # `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        # Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        # Apply the h_projection layer to this in order to compute init_decoder_hidden.
        # This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        # - `init_decoder_cell`:
        # `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        # Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        # Apply the c_projection layer to this in order to compute init_decoder_cell.
        # This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###
        # See the following docs, as you may need to use some of the following functions in your implementation:
        # Pack the padded sequence X before passing to the encoder:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        # Pad the packed sequence, enc_hiddens, returned by the encoder:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
        # Tensor Concatenation:
        # https://pytorch.org/docs/stable/torch.html#torch.cat
        # Tensor Permute:
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
        X = self.model_embeddings.source(source_padded)  # (src_len, b, e)
        X = pack_padded_sequence(input=X, lengths=source_lengths)  # (src_len, b, e)

        # enc_hiddens: (src_len, b, num_directions * h), where num_directions = 2
        # last_hidden: (num_layers * num_directions, b, h), where num_layers = 1
        # enc_hiddens: (num_layers * num_directions, b, h), where num_layers = 1
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
        # (src_len, b, 2*h) -> (src_len, b, 2*h)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens)
        # (src_len, b, 2*h) -> (b, src_len, 2*h)
        enc_hiddens = enc_hiddens.permute(1, 0, 2)

        # last_hidden: (2, b, h) -> (b, 2*h)
        # last_hidden: (b, 2*h) -> init_decoder_hidden: (b, h)
        init_decoder_hidden = self.h_projection(
            torch.cat((last_hidden[0], last_hidden[1]), 1))
        # last_cell: (2, b, h) -> (b, 2*h)
        # last_cell: (b, 2*h) -> init_decoder_hidden: (b, h)
        init_decoder_cell = self.c_projection(
            torch.cat((last_cell[0], last_cell[1]), 1))
        # dec_init_state: ((b, h), (b, h))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        # END YOUR CODE

        return enc_hiddens, dec_init_state

    def decode(self,
               enc_hiddens: torch.Tensor,
               enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor],
               target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens: (Tensor), Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks: (Tensor), Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state: (tuple(Tensor, Tensor)), Initial state and cell for decoder
        @param target_padded: (Tensor), Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size.

        @returns combined_outputs: (Tensor), combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        # YOUR CODE HERE (~9 Lines)
        # TODO:
        # 1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        # which should be shape (b, src_len, h),
        # where b = batch size, src_len = maximum source length, h = hidden size.
        # This is applying W_{attProj} to h^enc, as described in the PDF.
        #
        # 2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        # where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        #
        # 3. Use the torch.split function to iterate over the time dimension of Y.
        # Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        # - Squeeze Y_t into a tensor of dimension (b, e).
        # - Construct Ybar_t by concatenating Y_t with o_prev.
        # - Use the step function to compute the the Decoder's next (cell, state) values
        # as well as the new combined output o_t.
        # - Append o_t to combined_outputs
        # - Update o_prev to the new o_t.
        #
        # 4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        # tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        # where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        ###
        # Note:
        # - When using the squeeze() function make sure to specify the dimension you want to squeeze
        # over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        # Use the following docs to implement this functionality:
        # Zeros Tensor:
        # https://pytorch.org/docs/stable/torch.html#torch.zeros
        # Tensor Splitting (iteration):
        # https://pytorch.org/docs/stable/torch.html#torch.split
        # Tensor Dimension Squeezing:
        # https://pytorch.org/docs/stable/torch.html#torch.squeeze
        # Tensor Concatenation:
        # https://pytorch.org/docs/stable/torch.html#torch.cat
        # Tensor Stacking:
        # https://pytorch.org/docs/stable/torch.html#torch.stack

        # enc_hiddens: (b, src_len, h*2) -> enc_hiddens_proj: (b, h)
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        # target_padded: (tgt_len - 1, b) -> (tgt_len - 1, b, e)
        # the last <pad> token is chop out
        # because we do not input last <pad> token input
        # to guide the next token generation
        Y = self.model_embeddings.target(target_padded)

        for Y_t in torch.split(Y, 1, dim=0):
            # Y_t: (1, b, e) -> (b, e)
            Y_t = torch.squeeze(Y_t, dim=0)
            # o_prev: (b, h)
            # Ybar_t: (b, e + h)
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)
            # dec_state: ((b, h), (b, h))
            # combined_output: (b, h)
            dec_state, combined_output, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(combined_output)
            o_prev = combined_output

        # combined_outputs: #(tgt_len - 1) * (b, h) ->  (tgt_len - 1, b, h)
        combined_outputs = torch.stack(combined_outputs, dim=0)

        # END YOUR CODE

        return combined_outputs

    def step(self,
             Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t: (Tensor), Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state: (tuple(Tensor, Tensor)), Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens: (Tensor), Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj: (Tensor), Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks: (Tensor), Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        # YOUR CODE HERE (~3 Lines)
        # TODO:
        # 1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        # 2. Split dec_state into its two parts (dec_hidden, dec_cell)
        # 3. Compute the attention scores e_t, a Tensor shape (b, src_len).
        # Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        # Hints:
        # - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        # - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        # - Use batched matrix multiplication (torch.bmm) to compute e_t.
        # - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        # - When using the squeeze() function make sure to specify the dimension you want to squeeze
        # over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        # Use the following docs to implement this functionality:
        # Batch Multiplication:
        # https://pytorch.org/docs/stable/torch.html#torch.bmm
        # Tensor Unsqueeze:
        # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        # Tensor Squeeze:
        # https://pytorch.org/docs/stable/torch.html#torch.squeeze

        # Ybar_t: (b, e + h), dec_state: ((b, h), (b, h))
        # dec_state: ((b, h), (b, h))
        dec_state = self.decoder(Ybar_t, dec_state)
        (dec_hidden, dec_cell) = dec_state

        # enc_hiddens_proj: (b, src_len, h), dec_hidden: (b, h) -> (b, h, 1)
        # e_t: (b, src_len, 1) -> (b, src_len)
        e_t = torch.squeeze(
            torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, dim=2)), dim=2)

        # END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        # YOUR CODE HERE (~6 Lines)
        # TODO:
        # 1. Apply softmax to e_t to yield alpha_t
        # 2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        # attention output vector, a_t.
        # $$ Hints:
        #  - alpha_t is shape (b, src_len)
        # - enc_hiddens is shape (b, src_len, 2h)
        # - a_t should be shape (b, 2h)
        # - You will need to do some squeezing and unsqueezing.
        # Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        # 3. Concatenate dec_hidden with a_t to compute tensor U_t
        # 4. Apply the combined output projection layer to U_t to compute tensor V_t
        # 5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        # Use the following docs to implement this functionality:
        # Softmax:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        # Batch Multiplication:
        # https://pytorch.org/docs/stable/torch.html#torch.bmm
        # Tensor View:
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        # Tensor Concatenation:
        # https://pytorch.org/docs/stable/torch.html#torch.cat
        # Tanh:
        # https://pytorch.org/docs/stable/torch.html#torch.tanh

        # e_t: (b, src_len) -> alpha_t: (b, src_len)
        alpha_t = F.softmax(e_t, dim=1)

        # alpha_t: (b, src_len) -> (b, 1, src_len)
        # enc_hiddens: (b, src_len, 2*h)
        # a_t: (b, 1, 2*h) -> (b, 2*h)
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, dim=1), enc_hiddens),
                            dim=1)

        # dec_hidden: (b, h)
        # u_t: (b, 3*h)
        u_t = torch.cat([a_t, dec_hidden], dim=1)
        # v_t: (b, h)
        v_t = self.combined_output_projection(u_t)
        # O_t: (b, h)
        O_t = self.dropout(torch.tanh(v_t))

        # END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens: (Tensor), encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths: (List[int]), List of actual lengths for each of the sentences in the batch.

        @returns enc_masks: (Tensor), Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(
            0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str],
                    beam_size: int = 5,
                    max_decoding_time_step: int = 70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent: (List[str]), a single source sentence (words)
        @param beam_size: (int), beam size
        @param max_decoding_time_step: (int), maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        # src_encodings: (1, src_len, 2*h)
        # dec_init_vec: ((1, h), (1, h))
        src_encodings, dec_init_vec = self.encode(
            src_sents_var, [len(src_sent)])

        # src_encodings_att_linear: (1, src_len, h)
        src_encodings_att_linear = self.att_projection(src_encodings)

        # h_tm1: ((1, h), (1, h))
        h_tm1 = dec_init_vec
        # att_m1: (1, h)
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        # hyp_scores: (1,)
        hyp_scores = torch.zeros(
            len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            # exp_src_encodings: (hyp_num, src_len, 2*h)
            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            # exp_src_encodings_att_linear: (hyp_num, src_len, h)
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(
                                                                               1),
                                                                           src_encodings_att_linear.size(2))

            # y_tm1: (hyp_num,)
            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]]
                                  for hyp in hypotheses], dtype=torch.long, device=self.device)
            # y_t_embed: (hyp_num, e)
            y_t_embed = self.model_embeddings.target(y_tm1)

            # x: [(hyp_num, e) + (hyp_num, h)] = (hyp_num, e + h) ?
            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            # (h_t, cell_t): ((hyp_num, h), (hyp_num, h))
            # att_t: (hyp_num, h)
            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings,
                                                exp_src_encodings_att_linear,
                                                enc_masks=None)

            # log_p_t: (hyp_num, tgt_vocab_size)
            # log probabilities over target words
            log_p_t = F.log_softmax(
                self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            # contiuating_hyp_scores: (hyp_num,) -> (hyp_num, tgt_vocab_size) -> (hyp_num * tgt_vocab_size)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(
                1).expand_as(log_p_t) + log_p_t).view(-1)
            # top_cand_hyp_scores: (live_hyp_num,)
            # top_cand_hyp_pos: (live_hyp_num,)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(
                live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path: (str), path to model
        """
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path: (str), path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size,
                         hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class SGM(nn.Module):
    """ SGM (Sequence Generation Model for Multi-Label Classification)
        https://arxiv.org/pdf/1806.04822
        - Bidirectional LSTM Encoder
        - Unidirectional LSTM Decoder, highway network https://arxiv.org/pdf/1505.00387,
        - Attention, Global Embedding
    """

    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab,
                 dropout_rate=0.2,
                 loss_type='cross_entropy',
                 **kwargs):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(SGM, self).__init__()
        self.embeddings = Embeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        allow_loss_type = ['cross_entropy', 'focal_loss']

        if loss_type not in allow_loss_type:
            raise ValueError('loss_type must be one of `%s`, but got `%s`' % (allow_loss_type, loss_type))

        self.loss_type = loss_type

        # init variable for focal loss
        if self.loss_type == 'focal_loss':
            self.gamma = kwargs.pop('gamma', 2)

        # self.encoder (Bidirectional LSTM with bias)
        # self.h_projection, not mentioned in paper
        # self.c_projection, not mentioned in paper
        # self.decoder (LSTM Cell with bias)
        # self.h_att_projection (Linear Layer with no bias), called U_{a} in the paper.
        # self.s_att_projection (Linear Layer with no bias), called W_{a} in the paper.
        # self.att_projection (Linear Layer with no bias), called V_{a} in the paper.
        # self.s_output_projection (Linear Layer with no bias), called W_{d} in the paper.
        # self.c_output_projection (Linear Layer with no bias), called V_{d} in the paper.
        # self.target_vocab_projection (Linear Layer with no bias), called W_{o} in the PDF.
        # self.dropout (Dropout Layer)
        ###
        # Use the following docs to properly initialize these variables:
        # LSTM:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        # LSTM Cell:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        # Linear Layer:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        # Dropout Layer:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size,
                               bidirectional=True, bias=True)
        self.decoder = nn.LSTMCell(input_size=embed_size + 2 * hidden_size,
                                   hidden_size=hidden_size,
                                   bias=True)

        self.h_projection = nn.Linear(in_features=2 * hidden_size,
                                      out_features=hidden_size,
                                      bias=False)
        self.c_projection = nn.Linear(in_features=2 * hidden_size,
                                      out_features=hidden_size,
                                      bias=False)

        self.h_att_projection = nn.Linear(in_features=2 * hidden_size,
                                          out_features=hidden_size,
                                          bias=False)
        self.s_att_projection = nn.Linear(in_features=2 * hidden_size,
                                          out_features=hidden_size,
                                          bias=False)
        self.att_projection = nn.Linear(in_features=hidden_size,
                                        out_features=1,
                                        bias=False)

        self.s_output_projection = nn.Linear(in_features=hidden_size,
                                             out_features=hidden_size,
                                             bias=False)
        self.c_output_projection = nn.Linear(in_features=2 * hidden_size,
                                             out_features=hidden_size,
                                             bias=False)

        self.transform_gate = nn.Linear(in_features=hidden_size,
                                        out_features=hidden_size,
                                        bias=False)

        self.target_vocab_projection = nn.Linear(in_features=hidden_size,
                                                 out_features=len(vocab.tgt),
                                                 bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source: (List[List[str]]), list of source sentence tokens
        @param target: (List[List[str]]), list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores: (Tensor), a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(
            source, device=self.device)  # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(
            target, device=self.device)  # Tensor: (tgt_len, b)

        # Run the network forward:
        # 1. Apply the encoder to `source_padded` by calling `self.encode()`
        # 2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        # 3. Apply the decoder to compute combined-output by calling `self.decode()`
        # 4. Compute log probability distribution over the target vocabulary using the
        # combined_outputs returned by the `self.decode()` function.

        # 1. Apply the encoder to `source_padded` by calling `self.encode()`
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        # 2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        # 3. Apply the decoder to compute combined-output by calling `self.decode()`
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        # 4. Compute log probability distribution over the target vocabulary using the
        # combined_outputs: (tgt_len - 1, b, h) -> target_vocab_projection: (tgt_len - 1, b, tgt_vocab_size)
        # -> P: (tgt_len - 1, b, tgt_vocab_size)
        target_vocab_projection = self.target_vocab_projection(combined_outputs)
        # logits: (tgt_len - 1, b, tgt_vocab_size)
        logits = F.softmax(target_vocab_projection, dim=-1)

        return logits, target_padded

    def compute_loss(self,
                     logits: torch.Tensor,
                     target_padded: torch.Tensor):
        # target_logits: (tgt_len - 1, b, tgt_vocab_size)
        # target_padded: (tgt_len, b)

        # target_masks: (tgt_len, b)
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
        # P: (tgt_len - 1, b, tgt_vocab_size)
        P = torch.log(logits)

        if self.loss_type == 'cross_entropy':
            # loss: (tgt_len - 1, b)
            loss = -1 * torch.gather(P,
                                     index=target_padded[1:].unsqueeze(-1),
                                     dim=-1).squeeze(-1) * target_masks[1:]
        elif self.loss_type == 'focal_loss':
            # loss: (tgt_len - 1, b, tgt_vocab_size)
            loss = -1 * (1 - logits) ** self.gamma * P
            # loss: (tgt_len - 1, b)
            loss = torch.gather(loss,
                                index=target_padded[1:].unsqueeze(-1),
                                dim=-1).squeeze(-1) * target_masks[1:]

        batch_size = logits.size(1)
        loss = torch.sum(loss) / batch_size

        return loss

    def compute_micro_f1(self,
                         logits: torch.Tensor,
                         target_padded: torch.Tensor) -> float:

        def sentence_ids_to_multi_ones_hot_vector(y: List[int]) -> np.array:
            total_length = len(self.vocab.tgt)
            ones_hot = np.zeros(total_length, dtype=np.int)
            hot_indices = y
            ones_hot[hot_indices] = 1
            return ones_hot

        def sentences_ids_to_multi_ones_hot_vectors(ys: List[List[int]]) -> np.array:
            return np.array([sentence_ids_to_multi_ones_hot_vector(y) for y in ys],
                            dtype=np.int)

        # logits: (tgt_len - 1, b, tgt_vocab_size)
        # target_padded: (tgt_len, b)

        # target_padded: (tgt_len, b) -> (tgt_len - 1, b)
        target_padded = target_padded[1:]

        # pred: (tgt_len - 1, b)
        pred = torch.argmax(logits, dim=2)

        # convert to list
        # target_padded: (tgt_len - 1, b)
        # (tgt_len - 1, b)
        target_padded = target_padded.tolist()
        pred = pred.tolist()

        target_ones_hot_vectors = sentences_ids_to_multi_ones_hot_vectors(target_padded)
        pred_ones_hot_vectors = sentences_ids_to_multi_ones_hot_vectors(pred)

        micro_f1 = metrics.f1_score(target_ones_hot_vectors, pred_ones_hot_vectors, average='micro')

        return micro_f1

    def encode(self,
               source_padded: torch.Tensor,
               source_lengths: List[int]) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded: (Tensor), Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths: (List[int]), List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens: (Tensor), Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state: (tuple(Tensor, Tensor)), Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """

        # 1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        # src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        # that there is no initial hidden state or cell for the decoder.
        #
        # 2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        # - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        # - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        # - Note that the shape of the tensor returned by the encoder is (src_len, b, h*2) and we want to
        # return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        #
        # 3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        # - `init_decoder_hidden`:
        # `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        # Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        # Apply the h_projection layer to this in order to compute init_decoder_hidden.
        # This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        # - `init_decoder_cell`:
        # `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        # Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        # Apply the c_projection layer to this in order to compute init_decoder_cell.
        # This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###
        # See the following docs, as you may need to use some of the following functions in your implementation:
        # Pack the padded sequence X before passing to the encoder:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        # Pad the packed sequence, enc_hiddens, returned by the encoder:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
        # Tensor Concatenation:
        # https://pytorch.org/docs/stable/torch.html#torch.cat
        # Tensor Permute:
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
        X = self.embeddings.source(source_padded)  # (src_len, b, e)
        X = pack_padded_sequence(input=X, lengths=source_lengths)  # (src_len, b, e)

        # enc_hiddens: (src_len, b, num_directions * h), where num_directions = 2
        # last_hidden: (num_layers * num_directions, b, h), where num_layers = 1
        # enc_hiddens: (num_layers * num_directions, b, h), where num_layers = 1
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
        # (src_len, b, 2*h) -> (src_len, b, 2*h)
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens)
        # (src_len, b, 2*h) -> (b, src_len, 2*h)
        enc_hiddens = enc_hiddens.permute(1, 0, 2)

        # last_hidden: (2, b, h) -> (b, 2*h)
        # last_hidden: (b, 2*h) -> init_decoder_hidden: (b, h)
        init_decoder_hidden = self.h_projection(
            torch.cat((last_hidden[0], last_hidden[1]), 1))
        # last_cell: (2, b, h) -> (b, 2*h)
        # last_cell: (b, 2*h) -> init_decoder_hidden: (b, h)
        init_decoder_cell = self.c_projection(
            torch.cat((last_cell[0], last_cell[1]), 1))
        # dec_init_state: ((b, h), (b, h))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state

    def decode(self,
               enc_hiddens: torch.Tensor,
               enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor],
               target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens: (Tensor), Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks: (Tensor), Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state: (tuple(Tensor, Tensor)), Initial state and cell for decoder
        @param target_padded: (Tensor), Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size.

        @returns combined_outputs: (Tensor), combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        c_prev = torch.zeros(batch_size, 2 * self.hidden_size, device=self.device)
        combined_output_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        # 1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        # which should be shape (b, src_len, h),
        # where b = batch size, src_len = maximum source length, h = hidden size.
        # This is applying W_{attProj} to h^enc, as described in the PDF.
        #
        # 2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        # where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        #
        # 3. Use the torch.split function to iterate over the time dimension of Y.
        # Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        # - Squeeze Y_t into a tensor of dimension (b, e).
        # - Construct Ybar_t by concatenating Y_t with c_prev.
        # - Use the step function to compute the the Decoder's next (cell, state) values
        # as well as the new combined output o_t.
        # - Append o_t to combined_outputs
        # - Update c_prev to the new o_t.
        #
        # 4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        # tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        # where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        ###
        # Note:
        # - When using the squeeze() function make sure to specify the dimension you want to squeeze
        # over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        # Use the following docs to implement this functionality:
        # Zeros Tensor:
        # https://pytorch.org/docs/stable/torch.html#torch.zeros
        # Tensor Splitting (iteration):
        # https://pytorch.org/docs/stable/torch.html#torch.split
        # Tensor Dimension Squeezing:
        # https://pytorch.org/docs/stable/torch.html#torch.squeeze
        # Tensor Concatenation:
        # https://pytorch.org/docs/stable/torch.html#torch.cat
        # Tensor Stacking:
        # https://pytorch.org/docs/stable/torch.html#torch.stack

        # enc_hiddens: (b, src_len, h*2) -> enc_hiddens_proj: (b, h)
        enc_hiddens_proj = self.h_att_projection(enc_hiddens)
        # target_padded: (tgt_len - 1, b) -> (tgt_len - 1, b, e)
        # the last <pad> token is chop out
        # because we do not input last <pad> token input
        # to guide the next token generation
        Y = self.embeddings.target(target_padded)

        # count for avg embedding

        for Y_t in torch.split(Y, 1, dim=0):
            # Y_t: (1, b, e) -> (b, e)
            Y_t = torch.squeeze(Y_t, dim=0)

            # c_prev: (b, 2 * h)
            # Ybar_t: (b, e + h)
            Ybar_t = torch.cat((Y_t, c_prev), dim=1)
            # dec_state: ((b, h), (b, h))
            # combined_output: (b, h)
            dec_state, combined_output, c_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            transform_gate = torch.sigmoid(self.transform_gate(combined_output))
            combined_output = transform_gate * combined_output + (1 - transform_gate) * combined_output_prev
            combined_outputs.append(combined_output)
            c_prev = c_t
            combined_output_prev = combined_output

        # combined_outputs: #(tgt_len - 1) * (b, h) ->  (tgt_len - 1, b, h)
        combined_outputs = torch.stack(combined_outputs, dim=0)

        return combined_outputs

    def step(self,
             Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t: (Tensor), Concatenated Tensor of [Y_t c_prev], with shape (b, e + 2 * h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state: (tuple(Tensor, Tensor)), Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens: (Tensor), Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj: (Tensor), Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks: (Tensor), Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, 2*h), where b = batch size, h = hidden size.
        @returns c_t (Tensor): Tensor of shape (b, 2*h). Context vector, h = hidden_size
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        # YOUR CODE HERE (~3 Lines)
        # TODO:
        # 1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        # 2. Split dec_state into its two parts (dec_hidden, dec_cell)
        # 3. Compute the attention scores e_t, a Tensor shape (b, src_len).
        # Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        # Hints:
        # - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        # - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        # - Use batched matrix multiplication (torch.bmm) to compute e_t.
        # - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        # - When using the squeeze() function make sure to specify the dimension you want to squeeze
        # over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        # Use the following docs to implement this functionality:
        # Batch Multiplication:
        # https://pytorch.org/docs/stable/torch.html#torch.bmm
        # Tensor Unsqueeze:
        # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        # Tensor Squeeze:
        # https://pytorch.org/docs/stable/torch.html#torch.squeeze

        # Ybar_t: (b, e + h), dec_state: ((b, h), (b, h))
        # dec_state: ((b, h), (b, h))
        dec_state = self.decoder(Ybar_t, dec_state)
        (dec_hidden, dec_cell) = dec_state

        # s_att_projection: (b, h)
        s_att_proj = self.s_att_projection(torch.cat([dec_hidden, dec_cell], dim=1))

        # s_att_projection: (b, h) -> (b, 1, h), enc_hiddens_proj: (b, src_len, h)
        # combined_hidden_proj: (b, src_len, h)
        combined_hiddens_proj = torch.unsqueeze(s_att_proj, dim=1) + enc_hiddens_proj
        combined_hiddens_proj = torch.tanh(combined_hiddens_proj)
        # combined_hidden_proj: (b, src_len, h) -> (b, src_len, 1)
        # e_t: (b, src_len)
        combined_hiddens_proj = self.att_projection(combined_hiddens_proj)
        e_t = torch.squeeze(combined_hiddens_proj, dim=2)

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        # 1. Apply softmax to e_t to yield alpha_t
        # 2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        # attention output vector, c_t.
        # $$ Hints:
        #  - alpha_t is shape (b, src_len)
        # - enc_hiddens is shape (b, src_len, 2h)
        # - c_t should be shape (b, 2h)
        # - You will need to do some squeezing and unsqueezing.
        # Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        # 3. Concatenate dec_hidden with c_t to compute tensor U_t
        # 4. Apply the combined output projection layer to U_t to compute tensor V_t
        # 5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        # Use the following docs to implement this functionality:
        # Softmax:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        # Batch Multiplication:
        # https://pytorch.org/docs/stable/torch.html#torch.bmm
        # Tensor View:
        # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        # Tensor Concatenation:
        # https://pytorch.org/docs/stable/torch.html#torch.cat
        # Tanh:
        # https://pytorch.org/docs/stable/torch.html#torch.tanh

        # e_t: (b, src_len) -> alpha_t: (b, src_len)
        alpha_t = F.softmax(e_t, dim=1)

        # alpha_t: (b, src_len) -> (b, 1, src_len)
        # enc_hiddens: (b, src_len, 2*h)
        # c_t: (b, 1, 2*h) -> (b, 2*h)
        c_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, dim=1), enc_hiddens),
                            dim=1)

        # combined_output: (b, h)
        combined_output = self.s_output_projection(s_att_proj) + self.c_output_projection(c_t)
        combined_output = torch.tanh(combined_output)

        return dec_state, combined_output, c_t, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens: (Tensor), encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths: (List[int]), List of actual lengths for each of the sentences in the batch.

        @returns enc_masks: (Tensor), Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(
            0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str],
                    beam_size: int = 5,
                    max_decoding_time_step: int = 70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent: (List[str]), a single source sentence (words)
        @param beam_size: (int), beam size
        @param max_decoding_time_step: (int), maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(
            src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.h_att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, 2 * self.hidden_size, device=self.device)
        att_t_prev = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(
            len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0

        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(
                                                                               1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]]
                                  for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, c_t, _ = self.step(x, h_tm1,
                                                     exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            transform_gate = torch.sigmoid(self.transform_gate(att_t))
            att_t = transform_gate * att_t + (1 - transform_gate) * att_t_prev

            # log probabilities over target words
            log_p_t = F.log_softmax(
                self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(
                1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(
                live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = c_t[live_hyp_ids]
            att_t_prev = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path: (str), path to model
        """
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = SGM(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path: (str), path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embeddings.embed_size,
                         hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


model_map = {
    'NMT': NMT,
    'SGM': SGM,
}


def get_model(name: str):
    """get model by name.
    :param name:  model's name, must in the key of model_map
    :return: model class
    """
    if name in model_map:
        return model_map[name]
    raise ValueError('expect model name in %s, but got unknown model name: `%s`' % (model_map.keys(), name))


__all__ = ['get_model', 'Hypothesis']
