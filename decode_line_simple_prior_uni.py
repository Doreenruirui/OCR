# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import random
import string

import numpy as np
from six.moves import xrange
import tensorflow as tf
from os.path import join as pjoin

import nlc_model_prior as nlc_model
#import nlc_model_global as nlc_model
import nlc_data
#import nlc_data_no_filter as nlc_data
from util import initialize_vocabulary, get_tokenizer
from multiprocessing import Pool
import pdb

from scipy.signal import gaussian as gs
from flag import FLAGS
import re

def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

reverse_vocab, vocab = None, None


def create_model(session, vocab_size, forward_only):
    model = nlc_model.NLCModel(
        vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout,
        weight=FLAGS.weight, forward_only=True)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def tokenize(sent, vocab, depth=FLAGS.num_layers):
    align = pow(2, depth - 1)
    token_ids = nlc_data.sentence_to_token_ids(remove_nonascii(sent), vocab, get_tokenizer(FLAGS.tokenizer))
    ones = [1] * len(token_ids)
    pad = (align - len(token_ids)) % align

    token_ids += [nlc_data.PAD_ID] * pad
    ones += [0] * pad

    source = np.array(token_ids).reshape([-1, 1])
    mask = np.array(ones).reshape([-1, 1])

    return source, mask


def detokenize(sents, reverse_vocab):
    # TODO: char vs word
    def detok_sent(sent):
        outsent = ''
        for t in sent:
            if t >= len(nlc_data._START_VOCAB):
                outsent += reverse_vocab[t]
        return outsent
    return [detok_sent(s) for s in sents]


def decode_beam(model, sess, encoder_output, probs, max_beam_size, len_input):
    toks, probs, prob_trans = model.decode_beam(sess, encoder_output, probs, max_beam_size, len_input)
    return toks.tolist(), probs.tolist(), prob_trans


def decode_lattice(beamstrs, probs, prob_trans):
    num_str = len(beamstrs)
    dict_str = {}
    for i in range(num_str):
        cur_str = beamstrs[i]
        for j in range(0, len(cur_str)):
            prefix = cur_str[:j]
            if prefix not in dict_str:
                dict_str[prefix] = {}
                dict_str[prefix][cur_str[j]] = prob_trans[i][j + 1]
            else:
                dict_str[prefix][cur_str[j]] = prob_trans[i][j + 1]
        if cur_str not in dict_str:
            dict_str[cur_str] = {}
            dict_str[cur_str]['<eos>'] = prob_trans[i][len(cur_str) + 1]
            dict_str[cur_str + '<eos>'] = probs[i]
    return dict_str


def fix_sent(model, sess, sent):
    # Tokenize
    input_toks, mask = tokenize(sent, vocab)
    # Encode
    encoder_output = model.encode(sess, input_toks, mask)

    len_input = encoder_output.shape[0]
    probs_v = np.zeros(len_input + 13 + 1 + 13 + len_input)
    probs_v[len_input + 13 - int(FLAGS.variance): len_input + 13 + int(FLAGS.variance) + 1] = 1
    probs_v = np.exp(probs_v * FLAGS.scalar)/ sum(np.exp(probs_v * FLAGS.scalar))
    len_input = sum(mask)
    # Decode
    model.weight=FLAGS.weight
    beam_toks, probs, prob_trans = decode_beam(model, sess, encoder_output, probs_v, FLAGS.beam_size, len_input)
    # De-tokenize
    beam_strs = detokenize(beam_toks, reverse_vocab)

    # model.weight = 1
    # probs_v2 = np.ones(probs_v.shape[0]) * (1. / probs_v.shape[0])
    # beam_toks2, probs2, prob_trans2 = decode_beam(model, sess, encoder_output, probs_v2, FLAGS.beam_size, len_input)
    # beam_strs2 = detokenize(beam_toks2, reverse_vocab)

    return beam_strs, probs


def decode():
    # Prepare NLC data.
    global reverse_vocab, vocab

    folder_out = FLAGS.out_dir
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    print("Preparing NLC data in %s" % FLAGS.data_dir)
    vocab_path = pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, reverse_vocab = initialize_vocabulary(vocab_path)
    vocab_size = len(vocab)
    print("Vocabulary size: %d" % vocab_size)
    if FLAGS.gpu_frac == 1:
        sess = tf.Session()
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_frac)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, vocab_size, False)
    tic = time.time()
    with open(pjoin(FLAGS.data_dir, FLAGS.dev + '.x.txt'), 'r') as f_:
        lines = [ele.strip('\n') for ele in f_.readlines()]
    f_o = open(pjoin(folder_out, FLAGS.dev + '.prior.uni.o.txt.' + str(FLAGS.start) + '_' + str(FLAGS.end)), 'w')
    f_p = open(pjoin(folder_out, FLAGS.dev + '.prior.uni.p.txt.' + str(FLAGS.start) + '_' + str(FLAGS.end)), 'w')
    for line_id in range(FLAGS.start, FLAGS.end):
        line = lines[line_id].split('\t')[0]
        sent = line.strip()
        if len(sent) == 0:
            f_o.write('\n' * 100)
            continue
        output_sents, output_probs = fix_sent(model, sess, sent)
        #output_sents, output_probs = fix_sent(model, sess, sent.replace('-', '_'))
        for i in range(len(output_sents)):
            sent = output_sents[i]
            #sent = output_sents[i].replace('_', '-')
            prob = output_probs[i]
            f_o.write(sent + '\n')
            f_p.write(str(prob) + '\n')
            # f_o.write(sent + '\t' + str(prob) + '\n')
        if line_id % 100 == 0:
            toc = time.time()
            print(toc - tic)
            tic = time.time()
    f_o.close()
    f_p.close()


def main(_):
    decode()


if __name__ == "__main__":
    tf.app.run()
