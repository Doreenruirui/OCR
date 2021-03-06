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

import os
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
from multiprocessing import Pool
from os.path import join as pjoin
import nlc_model_multiple_weight as nlc_model
import nlc_data
from levenshtein import align, align_one2many
from util import initialize_vocabulary, get_tokenizer
import kenlm
import re
from flag import FLAGS

reverse_vocab, vocab, lm = None, None, None


def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', ' ', text)


def get_string_to_score(sent):
    sent = remove_nonascii(sent)
    items = []
    for ele in sent:
        if len(ele.strip()) == 0:
            items.append('<space>')
        else:
            items.append(ele)
    return ' '.join(items)


def score_sent(sents):
    global lm
    if 'low' in FLAGS.lmfile1:
        sents = [ele.lower() for ele in sents]
    score = np.zeros((len(sents), 1, 1))
    for i in range(len(sents)):
        score[i, 0, 0] = lm.score(get_string_to_score(sents[i]))
    return score


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


def padded(tokens, depth):
  maxlen = max(map(lambda x: len(x), tokens))
  align = pow(2, depth - 1)
  padlen = maxlen + (align - maxlen) % align
  return map(lambda token_list: token_list + [0] * (padlen - len(token_list)), tokens)


def tokenize(sents, vocab, depth=FLAGS.num_layers):
    token_ids = []
    for sent in sents:
        token_ids.append(nlc_data.sentence_to_token_ids(sent, vocab, get_tokenizer(FLAGS.tokenizer)))
    token_ids = padded(token_ids, depth)
    source = np.array(token_ids).T
    source_mask = (source != 0).astype(np.int32)
    return source, source_mask


def detokenize(sents, reverse_vocab):
    # TODO: char vs word
    def detok_sent(sent):
        outsent = ''
        for t in sent:
            if t >= len(nlc_data._START_VOCAB):
                outsent += reverse_vocab[t]
        return outsent
    return [detok_sent(s) for s in sents]


def decode_beam(model, sess, encoder_output, score, max_beam_size, len_input):
    toks, probs, prob_trans = model.decode_beam(sess, encoder_output, score, max_beam_size, len_input)
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
    scalar = FLAGS.alpha
    input_toks, mask = tokenize(sent, vocab)
    # score = score_sent(sent)
    # for i in range(score.shape[0]):
    #     score[i, 0, 0] = np.float_power(10, score[i, 0, 0] * scalar)
    num_sent = len(sent)
    score = np.zeros((num_sent, 1, 1))
    score[0, 0, 0] = 0.6
    for i in range(1, num_sent):
        score[i, 0, 0] = (1 - score[0, 0, 0]) / (num_sent - 1)
    # score = score / sum(sum(score))
    # Encode
    encoder_output = model.encode(sess, input_toks, mask)
    s1, s2, s3 = encoder_output.shape
    encoder_output = np.transpose(encoder_output, (1, 0, 2))
    encoder_output = np.reshape(encoder_output, [s2, s1, 1, s3])
    len_input = mask.shape[0]
    # Decode
    beam_toks, probs, prob_trans = decode_beam(model, sess, encoder_output, score, FLAGS.beam_size, len_input)
    # De-tokenize
    beam_strs = detokenize(beam_toks, reverse_vocab)

    return beam_strs, probs


def decode():
    # Prepare NLC data.
    global reverse_vocab, vocab, lm
    lm = kenlm.LanguageModel(pjoin(FLAGS.lmfile1, 'train.arpa'))
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
    model = create_model(sess, vocab_size, True)
    tic = time.time()
    with open(pjoin(FLAGS.data_dir, FLAGS.dev + '.x.txt'), 'r') as f_:
        lines = [ele.strip() for ele in f_.readlines()]
    flag_evl = 0
    if flag_evl:
        with open(pjoin(FLAGS.data_dir, FLAGS.dev + '.y.txt'), 'r') as f_:
            truths = [ele.lower().strip() for ele in f_.readlines()]
        f_o = open(pjoin(folder_out, FLAGS.dev + '.avg.' + '.ec.txt.' + str(FLAGS.start) + '_' + str(FLAGS.end)), 'w')
        pool = Pool(100)
    else:
        #f_o = open(pjoin(folder_out, FLAGS.dev + '.om1.txt.' + str(FLAGS.start) + '_' + str(FLAGS.end))    , 'w')
        f_o = open(pjoin(folder_out, FLAGS.dev + '.avg' + '.o.txt.' + str(FLAGS.start) + '_' + str(FLAGS.end)), 'w')
    for line_id in range(FLAGS.start, FLAGS.end):
        line = lines[line_id]
        if flag_evl:
            cur_truth = truths[line_id]
        sents = [ele for ele in line.strip('\n').split('\t')][:50]
        sents = [ele for ele in sents if len(ele.strip()) > 0]
        #sents = [ele.replace('-', '_') for ele in sents if len(ele.strip()) > 0]
        if len(sents) > 0:
            output_sents, output_probs = fix_sent(model, sess, sents)
            #output_sents = [ele.replace('_', '-') for ele in output_sents]
            if flag_evl:
                output_sents = [ele.lower() for ele in output_sents]
                best_dis, best_str = align_one2many(pool, cur_truth, output_sents)
                top_dis = align(cur_truth, output_sents[0])
                f_o.write('\t'.join(map(str, [best_dis, top_dis, len(cur_truth)])) + '\n')
            else:
                f_o.write('\n'.join(output_sents) + '\n')
        else:
            if flag_evl:
                f_o.write('\t'.join(map(str, [len(cur_truth), len(cur_truth), len(cur_truth)])) + '\n')
            else:
                f_o.write('\n' * 100)
        if line_id % 100 == 0:
            toc = time.time()
            print(toc - tic)
            tic = time.time()
    f_o.close()


def main(_):
    decode()


if __name__ == "__main__":
    tf.app.run()
