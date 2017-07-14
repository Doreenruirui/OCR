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

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh, sigmoid

import nlc_data

from lib_tensor.ops import dynamic_rnn as dynamic_rnn_1


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert(False)
    return optfn


class GRUCellAttn(rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_output, embed_size, scope=None):
        self.hs = encoder_output
        self.embed_size = embed_size
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn1"):
                hs2d = tf.reshape(self.hs, [-1, num_units])
                phi_hs2d = tanh(rnn_cell._linear(hs2d, num_units, True, 1.0))
                self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))
            with vs.variable_scope("Input1"):
                hs2d_i = tf.reshape(self.hs, [-1, num_units])
                phi_hs2d_i = tanh(rnn_cell._linear(hs2d_i, num_units, True, 1.0))
                self.phi_hs_i = tf.reshape(phi_hs2d_i, tf.shape(self.hs))
        super(GRUCellAttn, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        batch_size = tf.shape(inputs)[0]
        cur_input = tf.slice(inputs, [0, 0], [-1, self.embed_size])
        cur_indicate = tf.reshape(tf.transpose(tf.slice(inputs, [0, self.embed_size], [-1, -1]), [1, 0]),
                                  [-1, batch_size, 1])
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Input2"):
                gamma_s = tanh(rnn_cell._linear(state, self._num_units, True, 1.0))
            weights_s = tf.reduce_sum(self.phi_hs_i * gamma_s, reduction_indices=2, keep_dims=True)
            weights_s = tf.exp(weights_s - tf.reduce_max(weights_s,
                                                         reduction_indices=0,
                                                         keep_dims=True)) \
                        * tf.to_float(cur_indicate)
            weights_s /= (1e-6 + tf.reduce_sum(weights_s, reduction_indices=0, keep_dims=True))
            context_s = tf.reduce_sum(self.hs * weights_s, reduction_indices=0)
        new_inputs = tf.concat(1, [cur_input, context_s])
        gru_out, gru_state = super(GRUCellAttn, self).__call__(new_inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                gamma_h = tanh(rnn_cell._linear(gru_out, self._num_units, True, 1.0))
            weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
            weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
            weights /= (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True))
            context = tf.reduce_sum(self.hs * weights, reduction_indices=0)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
            return (out, out)


class NLCModel(object):
    def __init__(self, vocab_size, size, num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, dropout, forward_only=False, optimizer="adam"):
        self.size = size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.keep_prob_config = 1.0 - dropout
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        self.keep_prob = tf.placeholder(tf.float32)
        self.source_tokens = tf.placeholder(tf.int32, shape=[None, None])
        self.target_tokens = tf.placeholder(tf.int32, shape=[None, None])
        self.source_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.target_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.beam_size = tf.placeholder(tf.int32)
        self.target_length = tf.reduce_sum(self.target_mask, reduction_indices=0)
        self.len_input = tf.placeholder(tf.int32)

        self.decoder_state_input, self.decoder_state_output = [], []
        for i in xrange(num_layers):
            self.decoder_state_input.append(tf.placeholder(tf.float32, shape=[None, size]))

        with tf.variable_scope("NLC", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            # self.setup_globalattention()
            self.setup_encoder()
            self.set_input()
            self.setup_decoder()
            self.setup_loss()
            self.setup_beam()

        params = tf.trainable_variables()
        if not forward_only:
            opt = get_optimizer(optimizer)(self.learning_rate)
            self.gradients = tf.gradients(self.losses, params)
            clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, max_gradient_norm)
            self.gradient_norm = tf.global_norm(self.gradients)
            self.param_norm = tf.global_norm(params)
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)


    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            self.L_enc = tf.get_variable("L_enc", [self.vocab_size, self.size])
            self.L_dec = tf.get_variable("L_dec", [self.vocab_size, self.size])
            self.encoder_inputs = embedding_ops.embedding_lookup(self.L_enc, self.source_tokens)
            self.decoder_inputs = embedding_ops.embedding_lookup(self.L_dec, self.target_tokens)


    def setup_globalattention(self):
        with vs.variable_scope("global_attention"):
            self.attn_v = tf.get_variable("Att", shape=(1, self.size))


    def setup_encoder(self):
        self.encoder_cell = rnn_cell.GRUCell(self.size)
        with vs.variable_scope("PryamidEncoder"):
            inp = self.encoder_inputs
            mask = self.source_mask
            out = None
            for i in xrange(self.num_layers):
                with vs.variable_scope("EncoderCell%d" % i) as scope:
                    srclen = tf.reduce_sum(mask, reduction_indices=0)
                    out, _ = self.bidirectional_rnn(self.encoder_cell, inp, srclen, scope=scope)
                    dropin, mask = self.downscale(out, mask)
                    inp = self.dropout(dropin)
            self.encoder_output = out


    def set_input(self):
        T2 = tf.shape(self.target_tokens)[0]
        batch_size = tf.shape(self.source_tokens)[1]
        T1 = tf.shape(self.source_tokens)[0]
        self.rep_a = tf.tile(tf.transpose(self.source_tokens, [1, 0]), [T2, 1])
        rep_b = tf.reshape(self.target_tokens, [-1, 1])
        indicate = tf.equal(self.rep_a, rep_b)
        self.indicate = tf.to_float(tf.reshape(indicate, [T2, batch_size, T1]))
        # indicate = tf.transpose(tf.reshape(indicate, [len_out, batch_size, len_in]), [1,0,2])


    def setup_decoder(self):
        if self.num_layers > 1:
            self.decoder_cell = rnn_cell.GRUCell(self.size)
        self.attn_cell = GRUCellAttn(self.size, self.encoder_output, self.size, scope="DecoderAttnCell")

        with vs.variable_scope("Decoder"):
            inp = self.decoder_inputs
            for i in xrange(self.num_layers - 1):
                with vs.variable_scope("DecoderCell%d" % i) as scope:
                    out, state_output = rnn.dynamic_rnn(self.decoder_cell, inp, time_major=True,
                                                        dtype=dtypes.float32, sequence_length=self.target_length,
                                                        scope=scope, initial_state=self.decoder_state_input[i])
                    inp = self.dropout(out)
                    self.decoder_state_output.append(state_output)

            with vs.variable_scope("DecoderAttnCell") as scope:
                new_inp = tf.concat(2, [inp, self.indicate])
                out, state_output = dynamic_rnn_1(self.attn_cell, new_inp, time_major=True,
                                                    dtype=dtypes.float32, sequence_length=self.target_length,
                                                    scope=scope, initial_state=self.decoder_state_input[i+1])
                self.decoder_output = self.dropout(out)
                self.decoder_state_output.append(state_output)


    def decoder_graph(self, decoder_inputs, decoder_state_input):
        decoder_output, decoder_state_output = None, []
        inp = decoder_inputs

        with vs.variable_scope("Decoder", reuse=True):
            for i in xrange(self.num_layers - 1):
                with vs.variable_scope("DecoderCell%d" % i) as scope:
                    inp, state_output = self.decoder_cell(inp, decoder_state_input[i])
                    decoder_state_output.append(state_output)

            with vs.variable_scope("DecoderAttnCell") as scope:
                decoder_output, state_output = self.attn_cell(inp, decoder_state_input[i+1])
                decoder_state_output.append(state_output)

        return decoder_output, decoder_state_output

    def setup_beam(self):
        time_0 = tf.constant(0)
        beam_seqs_0 = tf.constant([[nlc_data.SOS_ID]])
        beam_probs_0 = tf.constant([0.])
        cand_seqs_prob_0 = tf.constant([[0.]])
        beam_seqs_prob_0 = tf.constant([[0.]])
        cand_seqs_0 = tf.constant([[nlc_data.EOS_ID]])
        cand_probs_0 = tf.constant([-3e38])

        state_0 = tf.zeros([1, self.size])
        states_0 = [state_0] * self.num_layers

        # bases_0 = tf.ones((1, self.beam_size), dtype=tf.int32)
        # mods_0 = tf.ones((1, self.beam_size), dtype=tf.int32)

        #total_probs_0 = tf.ones((1, self.beam_size)) * -3e38
        # cur_probs_0 = tf.ones((1, self.beam_size)) * -3e38
        # cur_eos_probs_0 = tf.ones((1, self.beam_size)) * -3e38
        #eos_probs_0 = tf.ones((1, self.beam_size))  * -3e38

        def beam_cond(cand_probs, cand_seqs, time, beam_probs, beam_seqs, cand_seq_prob, beam_seq_prob, *states):
            return tf.logical_and(tf.reduce_max(beam_probs) >= tf.reduce_min(cand_probs), time < tf.reshape(self.len_input, ()) + 10)

        def beam_step(cand_probs, cand_seqs, time, beam_probs, beam_seqs, cand_seq_prob, beam_seq_prob, *states):
            batch_size = tf.shape(beam_probs)[0]
            inputs = tf.reshape(tf.slice(beam_seqs, [0, time], [batch_size, 1]), [batch_size])
            decoder_input = embedding_ops.embedding_lookup(self.L_dec, inputs)
            decoder_output, state_output = self.decoder_graph(decoder_input, states)

            with vs.variable_scope("Logistic", reuse=True):
                do2d = tf.reshape(decoder_output, [-1, self.size])
                with vs.variable_scope("Error", reuse=True):
                    logits2d = rnn_cell._linear(do2d, self.vocab_size, True, 1.0)
                logprobs2d = tf.nn.log_softmax(logits2d)

            total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
            total_probs_noEOS = tf.concat(1, [tf.slice(total_probs, [0, 0], [batch_size, nlc_data.EOS_ID]),
                                              tf.tile([[-3e38]], [batch_size, 1]),
                                              tf.slice(total_probs, [0, nlc_data.EOS_ID + 1],
                                                       [batch_size, self.vocab_size - nlc_data.EOS_ID - 1])])
            flat_total_probs = tf.reshape(total_probs_noEOS, [-1])
            flat_log_probs = tf.reshape(logprobs2d, [-1])

            beam_k = tf.minimum(tf.size(flat_total_probs), self.beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)

            next_bases = tf.floordiv(top_indices, self.vocab_size)
            next_mods = tf.mod(top_indices, self.vocab_size)

            ############ Output
            # cur_log_probs = tf.reshape(tf.gather(flat_log_probs, top_indices), [1, -1])
            # cur_log_probs = tf.pad(cur_log_probs, [[0,0], [0,self.beam_size - beam_k]])
            # next_cur = tf.concat(0, [cur_prob, cur_log_probs])
            #
            # bases_vec = tf.pad(tf.reshape(next_bases, [1, -1]),
            #                    [[0,0],[0, self.beam_size - beam_k]])
            # out_bases = tf.concat(0, [bases, bases_vec])
            # mods_vec = tf.pad(tf.reshape(next_mods, [1, -1]), [[0,0],[0, self.beam_size - beam_k]])
            # out_mods = tf.concat(0, [mods, mods_vec])
            #
            # cur_eos = tf.slice(logprobs2d, [0, nlc_data.EOS_ID], [batch_size, 1])
            # cur_eos = tf.pad(tf.reshape(cur_eos, [1, -1]), [[0,0], [0, self.beam_size - batch_size]])
            # next_eos_cur = tf.concat(0, [cur_eos_probs, cur_eos])
            #############

            next_states = [tf.gather(state, next_bases) for state in state_output]
            next_beam_seqs = tf.concat(1, [tf.gather(beam_seqs, next_bases),
                                           tf.reshape(next_mods, [-1, 1])])
            ############# output #########
            # cur_prob = tf.gather(flat_total_probs, top_indices)
            # prob_vector = tf.reshape(cur_prob, [1, -1])
            # prob_vector = tf.pad(prob_vector, [[0,0],[0, self.beam_size - beam_k]])
            # next_total = tf.concat(0, [total, prob_vector])
            #############

            cand_seqs_pad = tf.pad(cand_seqs, [[0, 0], [0, 1]])
            beam_seqs_EOS = tf.pad(beam_seqs, [[0, 0], [0, 1]])
            new_cand_seqs = tf.concat(0, [cand_seqs_pad, beam_seqs_EOS])
            EOS_probs = tf.slice(total_probs, [0, nlc_data.EOS_ID], [batch_size, 1])

            ############## output
            # eos_prob_vec = tf.reshape(EOS_probs, [1, -1])
            # eos_prob_vec = tf.pad(eos_prob_vec, [[0,0], [0, self.beam_size - batch_size]])
            # next_eos_prob = tf.concat(0, [eos_probs, eos_prob_vec])
            ###############

            new_cand_probs = tf.concat(0, [cand_probs, tf.reshape(EOS_probs, [-1])])

            cand_k = tf.minimum(tf.size(new_cand_probs), self.beam_size)
            next_cand_probs, next_cand_indices = tf.nn.top_k(new_cand_probs, k=cand_k)
            next_cand_seqs = tf.gather(new_cand_seqs, next_cand_indices)

            new_beam_seq_prob = tf.reshape(tf.gather(beam_seq_prob, next_bases), [beam_k, -1])
            cur_seq_prob = tf.reshape(tf.gather(flat_log_probs, top_indices), [beam_k, -1])
            next_beam_seq_prob = tf.concat(1, [new_beam_seq_prob, cur_seq_prob])
            cur_eos = tf.slice(logprobs2d, [0, nlc_data.EOS_ID], [batch_size, 1])
            cand_seq_prob_pad = tf.pad(cand_seq_prob, [[0, 0], [0, 1]])
            beam_seq_prob_pad = tf.concat(1, [tf.reshape(beam_seq_prob,
                                                         [batch_size, -1]),
                                              cur_eos])
            new_cand_seq_prob = tf.concat(0, [cand_seq_prob_pad, beam_seq_prob_pad])
            next_cand_seq_prob = tf.gather(new_cand_seq_prob, next_cand_indices)


            return [next_cand_probs, next_cand_seqs, time + 1, next_beam_probs, next_beam_seqs, next_cand_seq_prob, next_beam_seq_prob] + next_states

        var_shape = []
        # var_shape.append((total_probs_0, tf.TensorShape([None, None])))
        # var_shape.append((eos_probs_0, tf.TensorShape([None, None])))
        var_shape.append((cand_probs_0, tf.TensorShape([None,])))
        var_shape.append((cand_seqs_0, tf.TensorShape([None, None])))
        # var_shape.append((cur_probs_0, tf.TensorShape([None, None])))
        # var_shape.append((cur_eos_probs_0, tf.TensorShape([None, None])))
        # var_shape.append((bases_0, tf.TensorShape([None, None])))
        # var_shape.append((mods_0, tf.TensorShape([None, None])))
        var_shape.append((time_0, time_0.get_shape()))
        var_shape.append((beam_probs_0, tf.TensorShape([None,])))
        var_shape.append((beam_seqs_0, tf.TensorShape([None, None])))
        var_shape.append((cand_seqs_prob_0, tf.TensorShape([None,None])))
        var_shape.append((beam_seqs_prob_0, tf.TensorShape([None,None])))
        var_shape.extend([(state_0, tf.TensorShape([None, self.size])) for state_0 in states_0])
        loop_vars, loop_var_shapes = zip(* var_shape)
        # loop_vars = (self.all_probs, self.all_total_probs, self.all_bases, self.all_mods, self.all_states) + loop_vars
        self.loop_vars = loop_vars
        self.loop_var_shapes = loop_var_shapes
        # print(loop_var_shapes)
        ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, back_prop=False)
        #ret_vars = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_vars, shape_invariants=loop_var_shapes, back_prop=False)
        #    time, beam_probs, beam_seqs, cand_probs, cand_seqs, _ = ret_vars
        self.vars = ret_vars
        self.beam_output= ret_vars[1]
        self.beam_scores = ret_vars[0]
        self.beam_trans = ret_vars[5]


    def setup_loss(self):
        with vs.variable_scope("Logistic"):
            doshape = tf.shape(self.decoder_output)
            T1 = tf.shape(self.encoder_output)[0]
            T2, batch_size = doshape[0], doshape[1]
            do2d = tf.reshape(self.decoder_output, [-1, self.size])
            en2d = tf.reshape(self.encoder_output, [-1, self.size])
            with vs.variable_scope("Predict"):
                pre_logits2d = rnn_cell._linear(do2d, 1, True, 1.0)
            self.pre2d = tf.reshape(tf.nn.sigmoid(pre_logits2d), [T2 * batch_size, 1])
            with vs.variable_scope("Copy"):
                copy2d_trans = tf.reshape(tanh(rnn_cell._linear(en2d, self.size, True, 1.0)),
                                          [T1, batch_size, -1])
            copy2d_mask = copy2d_trans * tf.to_float(tf.reshape(self.source_mask,
                                                            [T1, batch_size, 1]))
            copy2d_out = tf.reshape(self.decoder_output, [T2, 1, batch_size, -1])
            copy2d = tf.transpose(tf.reduce_sum(copy2d_out * copy2d_mask, reduction_indices=3), [0, 2, 1])
            indices_1 = tf.reshape(tf.tile(tf.reshape(tf.range(0, T2, 1), [T2, 1, 1]), [1, batch_size, T1]), [-1])
            indices_2 = tf.reshape(tf.tile(tf.reshape(tf.range(0, batch_size, 1),
                                                      [1, batch_size, 1]),
                                           [T2, 1, T1]),
                                   [-1])
            indices_3 = tf.reshape(tf.transpose(tf.tile(tf.reshape(self.source_tokens,
                                                                   [1, T1, batch_size]),
                                                        [T2, 1, 1]),
                                                [0, 2, 1]),
                                   [-1])
            prob_shape = tf.reshape(T2 * batch_size * self.vocab_size, [1])
            copy_prob = tf.zeros(prob_shape, dtype=tf.float32)
            linear_indices = indices_1 * batch_size * self.vocab_size + indices_2 * self.vocab_size + indices_3
            flat_w_copy = tf.reshape(copy2d, [-1])
            unchanged_indices = tf.range(tf.size(copy_prob))
            flat_vocab_prob = tf.dynamic_stitch([unchanged_indices, linear_indices], [copy_prob, flat_w_copy])
            vocab_copy = tf.reshape(flat_vocab_prob, [T2 * batch_size, self.vocab_size])
            self.prob_copy = tf.nn.softmax(vocab_copy)
            with vs.variable_scope("Error"):
                vocab_error = rnn_cell._linear(do2d, self.vocab_size, True, 1.0)
            self.prob_error = tf.nn.softmax(vocab_error)
            self.total_prob = self.pre2d * self.prob_copy + (1 - self.pre2d) * self.prob_error
            outputs2d = tf.log(tf.clip_by_value(self.total_prob,1e-10,1.0))
            # max_copy = tf.reduce_max(vocab_copy, reduction_indices=1, keep_dims=True)
            # max_error = tf.reduce_max(vocab_error, reduction_indices=1, keep_dims=True)
            # max_v = tf.reduce_max(tf.concat(1, [max_copy, max_error]), reduction_indices=1, keep_dims=True)
            # prob_copy = tf.exp(vocab_copy - max_v)
            # prob_error = tf.exp(vocab_error - max_v)
            # vocab_prob = prob_copy + prob_error
            # vocab_prob /= (1e-6 + tf.reduce_sum(vocab_prob, reduction_indices=1, keep_dims=True))
            # outputs2d = tf.reshape(tf.log(vocab_prob + 1e-20), [-1, self.vocab_size])
            self.outputs = tf.reshape(outputs2d, tf.pack([T2, batch_size, self.vocab_size]))


            targets_no_GO = tf.slice(self.target_tokens, [1, 0], [-1, -1])
            masks_no_GO = tf.slice(self.target_mask, [1, 0], [-1, -1])
            labels1d = tf.reshape(tf.pad(targets_no_GO, [[0, 1], [0, 0]]), [-1])
            labels2d = tf.one_hot(labels1d, depth=self.vocab_size, on_value=1.0, off_value=0.0, axis=-1)
            mask1d = tf.reshape(tf.pad(masks_no_GO, [[0, 1], [0, 0]]), [-1])
            losses2d = -tf.reduce_sum(outputs2d * labels2d, reduction_indices=1) * tf.to_float(mask1d)
            self.losses = tf.reduce_sum(losses2d) / tf.to_float(batch_size)

    # def setup_loss(self):
    #     with vs.variable_scope("Logistic"):
    #         doshape = tf.shape(self.decoder_output)
    #         T, batch_size = doshape[0], doshape[1]
    #         do2d = tf.reshape(self.decoder_output, [-1, self.size])
    #         logits2d = rnn_cell._linear(do2d, self.vocab_size, True, 1.0)
    #         outputs2d = tf.nn.log_softmax(logits2d)
    #         self.outputs = tf.reshape(outputs2d, tf.pack([T, batch_size, self.vocab_size]))
    #
    #         targets_no_GO = tf.slice(self.target_tokens, [1, 0], [-1, -1])
    #         masks_no_GO = tf.slice(self.target_mask, [1, 0], [-1, -1])
    #         # easier to pad target/mask than to split decoder input since tensorflow does not support negative indexing
    #         labels1d = tf.reshape(tf.pad(targets_no_GO, [[0, 1], [0, 0]]), [-1])
    #         mask1d = tf.reshape(tf.pad(masks_no_GO, [[0, 1], [0, 0]]), [-1])
    #         losses1d = tf.nn.sparse_softmax_cross_entropy_with_logits(logits2d, labels1d) * tf.to_float(mask1d)
    #         losses2d = tf.reshape(losses1d, tf.pack([T, batch_size]))
    #         self.losses = tf.reduce_sum(losses2d) / tf.to_float(batch_size)

    def dropout(self, inp):
        return tf.nn.dropout(inp, self.keep_prob)

    def downscale(self, inp, mask):
        return inp, mask
        with vs.variable_scope("Downscale"):
            inshape = tf.shape(inp)
            T, batch_size, dim = inshape[0], inshape[1], inshape[2]
            inp2d = tf.reshape(tf.transpose(inp, perm=[1, 0, 2]), [-1, 2 * self.size])
            out2d = rnn_cell._linear(inp2d, self.size, True, 1.0)
            out3d = tf.reshape(out2d, tf.pack((batch_size, tf.to_int32(T/2), dim)))
            out3d = tf.transpose(out3d, perm=[1, 0, 2])
            out3d.set_shape([None, None, self.size])
            out = tanh(out3d)

            mask = tf.transpose(mask)
            mask = tf.reshape(mask, [-1, 2])
            mask = tf.cast(mask, tf.bool)
            mask = tf.reduce_any(mask, reduction_indices=1)
            mask = tf.to_int32(mask)
            mask = tf.reshape(mask, tf.pack([batch_size, -1]))
            mask = tf.transpose(mask)
        return out, mask

    def bidirectional_rnn(self, cell, inputs, lengths, scope=None):
        name = scope.name or "BiRNN"
        # Forward direction
        with vs.variable_scope(name + "_FW") as fw_scope:
            output_fw, output_state_fw = rnn.dynamic_rnn(cell, inputs, time_major=True, dtype=dtypes.float32,
                                                         sequence_length=lengths, scope=fw_scope)
        # Backward direction
        inputs_bw = tf.reverse_sequence(inputs, tf.to_int64(lengths), seq_dim=0, batch_dim=1)
        with vs.variable_scope(name + "_BW") as bw_scope:
            output_bw, output_state_bw = rnn.dynamic_rnn(cell, inputs_bw, time_major=True, dtype=dtypes.float32,
                                                         sequence_length=lengths, scope=bw_scope)

        output_bw = tf.reverse_sequence(output_bw, tf.to_int64(lengths), seq_dim=0, batch_dim=1)

        outputs = output_fw + output_bw
        output_state = output_state_fw + output_state_bw

        return (outputs, output_state)

    def set_default_decoder_state_input(self, input_feed, batch_size):
        default_value = np.zeros([batch_size, self.size])
        for i in xrange(self.num_layers):
            input_feed[self.decoder_state_input[i]] = default_value

    def feed_input(self, source_tokens, source_mask, target_tokens, target_mask):
        input_feed = {}
        input_feed[self.source_tokens] = source_tokens
        input_feed[self.target_tokens] = target_tokens
        input_feed[self.source_mask] = source_mask
        input_feed[self.target_mask] = target_mask
        input_feed[self.keep_prob] = self.keep_prob_config
        self.set_default_decoder_state_input(input_feed, target_tokens.shape[1])

    def train(self, session, source_tokens, source_mask, target_tokens, target_mask):
        input_feed = {}
        input_feed[self.source_tokens] = source_tokens
        input_feed[self.target_tokens] = target_tokens
        input_feed[self.source_mask] = source_mask
        input_feed[self.target_mask] = target_mask
        input_feed[self.keep_prob] = self.keep_prob_config
        self.set_default_decoder_state_input(input_feed, target_tokens.shape[1])
        output_feed = [self.updates, self.gradient_norm, self.losses,
                       self.param_norm, self.gradients]
        # output_feed = [self.updates, self.gradient_norm, self.losses,
        #                self.param_norm, self.gradients, self.pre2d, self.prob_copy, self.prob_error]

        outputs = session.run(output_feed, input_feed)

        return outputs[1], outputs[2], outputs[3], outputs[4]

    def test(self, session, source_tokens, source_mask, target_tokens, target_mask):
        input_feed = {}
        input_feed[self.source_tokens] = source_tokens
        input_feed[self.target_tokens] = target_tokens
        input_feed[self.source_mask] = source_mask
        input_feed[self.target_mask] = target_mask
        input_feed[self.keep_prob] = 1.
        self.set_default_decoder_state_input(input_feed, target_tokens.shape[1])

        output_feed = [self.losses]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def encode(self, session, source_tokens, source_mask):
        input_feed = {}
        input_feed[self.source_tokens] = source_tokens
        input_feed[self.source_mask] = source_mask
        input_feed[self.keep_prob] = 1.

        output_feed = [self.encoder_output]

        outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def decode(self, session, encoder_output, target_tokens, target_mask=None, decoder_states=None):
        input_feed = {}
        input_feed[self.encoder_output] = encoder_output
        input_feed[self.target_tokens] = target_tokens
        input_feed[self.target_mask] = target_mask if target_mask else np.ones_like(target_tokens)
        input_feed[self.keep_prob] = 1.

        if not decoder_states:
            self.set_default_decoder_state_input(input_feed, target_tokens.shape[1])
        else:
            for i in xrange(self.num_layers):
                input_feed[self.decoder_state_input[i]] = decoder_states[i]

        output_feed = [self.outputs] + self.decoder_state_output

        outputs = session.run(output_feed, input_feed)

        return outputs[0], None, outputs[1:]

    def decode_beam(self, session, encoder_output, beam_size, len_input):
        input_feed = {}
        input_feed[self.encoder_output] = encoder_output
        input_feed[self.keep_prob] = 1.
        input_feed[self.beam_size] = beam_size
        input_feed[self.len_input] = len_input

        output_feed = [self.beam_output, self.beam_scores, self.beam_trans]

        outputs = session.run(output_feed, input_feed)
        # output = session.run(self.vars, input_feed)
        # return outputs[0], outputs[1], output
        return outputs[0], outputs[1], outputs[2]
