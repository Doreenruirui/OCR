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


import nlc_model
import nlc_data
from util import get_tokenizer
from levenshtein import align_all, align
from multiprocessing import Pool
import kenlm


tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.1, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("size", 400, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
tf.app.flags.DEFINE_integer("max_seq_len", 200, "Maximum sequence length.")
tf.app.flags.DEFINE_integer("start", 0, "Decode from.")
tf.app.flags.DEFINE_integer("end", 0, "Decode to.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "Set to WORD to train word level model.")
tf.app.flags.DEFINE_integer("beam_size", 8, "Size of beam.")
tf.app.flags.DEFINE_integer("nthread", 8, "number of threads.")
tf.app.flags.DEFINE_string("lmfile1", None, "arpa file of the language model.")
tf.app.flags.DEFINE_string("lmfile2", None, "arpa file of the language model.")
tf.app.flags.DEFINE_float("alpha", 0, "Language model relative weight.")
tf.app.flags.DEFINE_float("beta", 0, "Language model relative weight.")
tf.app.flags.DEFINE_float("gpu_frac", 1, "GPU Fraction to be used.")

FLAGS = tf.app.flags.FLAGS
reverse_vocab, vocab = None, None
lm1 = None
lm2 = None


def create_model(session, vocab_size, forward_only):
    model = nlc_model.NLCModel(
        vocab_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, FLAGS.dropout,
        forward_only=forward_only)
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
    token_ids = nlc_data.sentence_to_token_ids(sent, vocab, get_tokenizer(FLAGS))
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


def lm_rank(strs, probs, lm):
    if lm is not None:
        lmscores = [lm.score(s)/(1+len(s.split())) for s in strs]
    else:
        lmscores = [0 for i in range(len(strs))]
    return lmscores


def sort_score_thread(para):
    thread_num, a, b, probs, score1, score2 = para
    rescores = [(1 - a - b) * q + a * l + b * p for (l, p, q) in zip(score1, score2, probs)]
    rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x: x[1])]
    generated = rerank[-1]
    return thread_num, generated


def lm_rank_all(P, strs, probs, list_para):
    lmscore1 = lm_rank(strs, probs, lm1)
    lmscore2 = lm_rank(strs, probs, lm2)
    probs = [p / (len(s)+1) for (s, p) in zip(strs, probs)]
    #for (s, p, l) in zip(strs, probs, lmscore1, lmscore2):
    #    print(s, p, l)
    paras = [(x, list_para[x][0], list_para[x][1], probs, lmscore1, lmscore2)
                 for x in range(len(list_para))]
    results = P.map(sort_score_thread, paras)
    result_str = ['' for i in range(len(list_para))]
    for thread_num, rank in results:
        result_str[thread_num] = strs[rank[-1]]
    return result_str


def decode_beam(model, sess, encoder_output, max_beam_size, len_input):
    toks, probs, prob_trans = model.decode_beam(sess, encoder_output, max_beam_size, len_input)
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


def fix_sent(P, model, sess, sent):
    list_para = []
    # Tokenize
    input_toks, mask = tokenize(sent, vocab)
    # Encode
    encoder_output = model.encode(sess, input_toks, mask)
    len_input = sum(mask)
    # Decode
    beam_toks, probs, prob_trans = decode_beam(model, sess, encoder_output, FLAGS.beam_size, len_input)
    # De-tokenize
    beam_strs = detokenize(beam_toks, reverse_vocab)

    # lattice = decode_lattice(beam_strs, probs, prob_trans)
    # Language Model ranking
    if lm1 is not None and lm2 is not None:
        best_str = lm_rank_all(P, beam_strs, probs, list_para)
    else:
        best_str = []
    return beam_strs, probs, best_str


def decode():
    # Prepare NLC data.
    global reverse_vocab, vocab, lm1, lm2

    if FLAGS.lmfile1 is not None:
      print("Loading Language model from %s" % FLAGS.lmfile1)
      lm1 = kenlm.LanguageModel(FLAGS.lmfile1)

    if FLAGS.lmfile2 is not None:
      print("Loading Language model from %s" % FLAGS.lmfile2)
      lm2 = kenlm.LanguageModel(FLAGS.lmfile2)

    print("Preparing NLC data in %s" % FLAGS.data_dir)
    nthread = FLAGS.nthread
    pool = Pool(processes=nthread)

    x_train, y_train, x_dev, y_dev, vocab_path = nlc_data.prepare_nlc_data(
        FLAGS.data_dir, FLAGS.max_vocab_size,
        tokenizer=get_tokenizer(FLAGS))
    vocab, reverse_vocab = nlc_data.initialize_vocabulary(vocab_path)
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
    with open(pjoin(FLAGS.data_dir, 'dev.x.txt'), 'r') as f_:
        lines = f_.readlines()
    with open(pjoin(FLAGS.data_dir, 'dev.y.txt'), 'r') as f_:
        truths = f_.readlines()
    folder_out = pjoin(FLAGS.data_dir, str(FLAGS.beam_size))
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    f_o = open(pjoin(folder_out, 'dev.o.txt.' + str(FLAGS.start) + '_' + str(FLAGS.end)), 'w')
    for line_id in range(FLAGS.start, FLAGS.end):
        line = lines[line_id]
        sent = line.strip()
        output_sents, output_probs, best_str = fix_sent(pool, model, sess, sent)
        min_dis, min_str = align_all(pool, truths[line_id], output_sents, nthread, flag_char=1)
        min_dis_word, min_str_word = align_all(pool, truths[line_id], output_sents, nthread, flag_char=0)
        f_o.write('\t'.join(best_str) + '\t' + min_str + '\t' + min_str_word + '\t' + output_sents[0] + '\n')
        print(line_id)
        if line_id % 100 == 0:
            toc = time.time()
            print(toc - tic)
            tic = time.time()
    f_o.close()


def main(_):
    decode()


if __name__ == "__main__":
    tf.app.run()
