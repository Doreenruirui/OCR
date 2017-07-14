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
import nlc_model_sample as nlc_model
from util import initialize_vocabulary, get_tokenizer
import util
from flag import FLAGS


reverse_vocab, vocab = None, None
lm = None


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
    token_ids = util.sentence_to_token_ids(sent, vocab, get_tokenizer(FLAGS.tokenizer))
    ones = [1] * len(token_ids)
    pad = (align - len(token_ids)) % align

    token_ids += [util.PAD_ID] * pad
    ones += [0] * pad

    source = np.array(token_ids).reshape([-1, 1])
    mask = np.array(ones).reshape([-1, 1])

    return source, mask


def detokenize(sents, reverse_vocab):
    # TODO: char vs word
    def detok_sent(sent):
        outsent = ''
        for t in sent:
            if t >= len(util._START_VOCAB):
                outsent += reverse_vocab[t]
        return outsent
    return [detok_sent(s) for s in sents]


def lm_rank(strs, probs):
    if lm is None:
        return strs[0]
    a = FLAGS.alpha
    lmscores = [lm.score(s)/(1+len(s.split())) for s in strs]
    probs = [ p / (len(s)+1) for (s, p) in zip(strs, probs) ]
    for (s, p, l) in zip(strs, probs, lmscores):
        print(s, p, l)

    rescores = [(1 - a) * p + a * l for (l, p) in zip(lmscores, probs)]
    rerank = [rs[0] for rs in sorted(enumerate(rescores), key=lambda x: x[1])]
    generated = strs[rerank[-1]]
    lm_score = lmscores[rerank[-1]]
    nw_score = probs[rerank[-1]]
    score = rescores[rerank[-1]]
    return generated



def decode_beam(model, sess, encoder_output, max_beam_size, len_input):
    toks, probs,_ = model.decode_beam(sess, encoder_output, max_beam_size, len_input)
    return toks.tolist(), probs.tolist()


def fix_sent(model, sess, sent):
    # Tokenize
    input_toks, mask = tokenize(sent, vocab)
    # Encode
    encoder_output = model.encode(sess, input_toks, mask)
    # Decode
    len_input = sum(mask)
    beam_toks, probs = decode_beam(model, sess, encoder_output, FLAGS.beam_size, len_input)
    # De-tokenize
    beam_strs = detokenize(beam_toks, reverse_vocab)
    return beam_strs, probs

def decode():
    global reverse_vocab, vocab, lm
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)
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
    print("Using %.2f percent of GPU" % FLAGS.gpu_frac)
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, vocab_size, False)
    line_id = 0
    tic = time.time()
    with open(pjoin(FLAGS.data_dir, FLAGS.dev + '.x.txt'), 'r') as f_:
        lines = f_.readlines()
    f_p = open(pjoin(FLAGS.out_dir, FLAGS.dev + '.p.txt.' + str(FLAGS.start) + '_' + str(FLAGS.end)), 'w')
    f_o = open(pjoin(FLAGS.out_dir, FLAGS.dev + '.o.txt.' + str(FLAGS.start) + '_' + str(FLAGS.end)), 'w')
    for line in lines[FLAGS.start:FLAGS.end]:
        line_id += 1
        print(line_id)
        if line_id % 100 == 0:
            toc = time.time()
            print(toc - tic)
            tic = time.time()
        sent = line.strip()
        output_sents, output_probs = fix_sent(model, sess, sent)
        for sent in output_sents:
            f_o.write(sent + '\n')
        for prob in output_probs:
            f_p.write('%.5f\n' % np.exp(prob))
    f_o.close()
    f_p.close()


def main(_):
    decode()


if __name__ == "__main__":
    tf.app.run()
