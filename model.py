"""Neural language correction model.

Reference:
  - https://arxiv.org/abs/1603.09727

"""

import numpy as np
from chainer import cuda, Variable, Chain
import chainer.links as L
import chainer.functions as F
from nlc.utils import VOC_SIZE

xp = None
try:
    xp = cuda.cupy
except AttributeError:
    xp = np


class EncoderDecoder(Chain):
    """Character-based neural language correction model with attention mechamism."""

    def __init__(self, voc_size, n_layer, size):

        self.voc_size = voc_size

        enc_hidden_size_list = [size for _ in range(n_layer)]
        dec_hidden_size_list = [size for _ in range(n_layer)]
        enc_pyr_out_size_list = [size for _ in range(n_layer)]
        c_out_size = size

        super(EncoderDecoder, self).__init__(
            enc=Encoder(n_layer, voc_size, enc_hidden_size_list, enc_pyr_out_size_list),
            dec=Decoder(n_layer, voc_size, dec_hidden_size_list, c_out_size),
            linear_last=L.Linear(dec_hidden_size_list[-1], voc_size)
        )

    def __call__(self, x, y, flatten=False):
        """Calculate feedfoward outputs before softmax activation function
        :param x: float32 variable of (x_max_seq_size, batch_size, voc_size)
        :param y: float32 variable of (y_max_seq_size, batch_size, voc_size)
        :return: float32 variable of (d_max_seq_size (= y_max_seq_size - 1), batch_size, voc_size)
        """

        c = self.enc(x)  # (c_max_seq_size, batch_size, c_out_size)
        d = self.dec(c, y)  # (d_max_seq_size, batch_size, d_out_size)

        d_max_seq_size, batch_size, d_out_size = d.shape

        d_flatten = F.reshape(d, (d_max_seq_size * batch_size, d_out_size))
        out_flatten = self.linear_last(d_flatten)  # (d_max_seq_size * batch_size, voc_size)
        if flatten:
            out = out_flatten
        else:
            out = F.reshape(out_flatten, (d_max_seq_size, batch_size, self.voc_size))  # (d_max_seq_size, batch_size, voc_size)

        return out

    def _lossfun(self, x, y):
        d_max_seq_size = y.shape[0] - 1
        batch_size = x.shape[1]

        out_flatten = self(x, y[:-1], flatten=True)  # (d_max_seq_size * batch_size, voc_size, voc_size)
        target_flatten = F.argmax(F.reshape(y[1:], (d_max_seq_size * batch_size, self.voc_size)), axis=1)  # (d_max_seq_size * batch_size, voc_size)

        return out_flatten, target_flatten

    def lossfun(self, x, y):
        """
        :param x: float32 variable of (x_max_seq_size, batch_size, voc_size)
        :param y: float32 variable of (y_max_seq_size, batch_size, voc_size)
        :return: scalar variable
        """
        out_flatten, target_flatten = self._lossfun(x, y)
        return F.softmax_cross_entropy(out_flatten, target_flatten)

    def argmax_likelihood(self, x, y, n):
        """Return indexes of samples having top n log-likelihood.
        :param x: float32 variable of (x_max_seq_size, voc_size)
        :param y: float32 variable of (y_max_seq_size, batch_size, voc_size)
        :return: list
        """
        y_seq_size, batch_size, voc_size = y.shape
        # duplicate same samples as many as batch_size of y
        x = F.tile(F.expand_dims(x, 1), (1, batch_size, 1))

        out_flatten = self(x, y[:-1], flatten=True)
        out_log_softmax = F.log_softmax(out_flatten)  # (y_max_seq_size * batch_size, voc_size)
        target_flatten = F.reshape(y[1:], ((y_seq_size - 1) * batch_size, voc_size))

        nll_flatten = - F.sum(out_log_softmax * target_flatten, axis=1)
        nll = F.sum(F.reshape(nll_flatten, ((y_seq_size - 1), batch_size)), axis=0) / y_seq_size  # (batch_size, )
        nll = cuda.to_cpu(nll.data)

        arg_nsmallest = list(np.argsort(nll)[:n])
        score_nsmallest = nll[arg_nsmallest]
        mean_score_nsmallest = list(score_nsmallest / (y_seq_size))

        return mean_score_nsmallest, arg_nsmallest

    def reset_state(self):
        self.enc.reset_state()
        self.dec.reset_state()


class Encoder(Chain):
    """Pyramidal bidirectional GRU.

    Eaxample:
    >>> from nlc.utils import make_batch_variable, VOC_SIZE
    >>> X = make_batch_variable(["I loves you .", "You loves me ."])
    >>> enc = Encoder(2, VOC_SIZE, [60, 30], [50, 20])
    >>> c = enc(X)
    >>> c.shape  # (max_seq_len / 2^2, batch_size, c_out_size)
    (4, 2, 20)
    """

    def __init__(self, n_layer, in_size, hidden_size_list, pyr_out_size_list):
        assert len(hidden_size_list) == n_layer
        assert len(pyr_out_size_list) == n_layer

        self.n_layer = n_layer
        super(Encoder, self).__init__()

        for i in range(self.n_layer):
            if i == 0:
                self.add_link('BidirectionalGRU_{}'.format(i),
                              BidirectionalGRU(in_size, hidden_size_list[0], pyr_out_size_list[0]))
                continue

            self.add_link('BidirectionalGRU_{}'.format(i),
                          BidirectionalGRU(pyr_out_size_list[i - 1], hidden_size_list[i], pyr_out_size_list[i]))

    def __call__(self, c):
        for i in range(self.n_layer):
            c = self['BidirectionalGRU_{}'.format(i)](c)

        return c

    def reset_state(self):
        for i in range(self.n_layer):
            self['BidirectionalGRU_{}'.format(i)].reset_state()


class Decoder(Chain):
    """GRU based decoder"""

    def __init__(self, n_layer, in_size, hidden_size_list, c_out_size):
        assert len(hidden_size_list) == n_layer

        self.n_layer = n_layer

        super(Decoder, self).__init__()

        for i in range(self.n_layer - 1):
            out_size = hidden_size_list[i]
            self.add_link('GRU_{}'.format(i), L.StatefulGRU(in_size, out_size))
            in_size = out_size

        self.add_link('GRUAtten'.format(self.n_layer - 1), GRUAtten(hidden_size_list[-2], hidden_size_list[-1], c_out_size))

    def __call__(self, c, d):
        """Calculate feed-forward outputs of decoder
        """
        return self.forward(c, d)

    def forward(self, c, d):
        """Calculate feed-forward of d"""

        def forward_per_layer(ith_layer, d):
            seq_size = d.shape[0]

            d_list = []
            for t in range(seq_size):
                d_t_new = self['GRU_{}'.format(ith_layer)](d[t])  # (batch_size, hidden_size)
                d_list.append(d_t_new)

            d_new = F.concat([F.expand_dims(d_t, 0) for d_t in d_list], 0)  # (seq_size, batch_size, hidden_size)

            return d_new

        for i in range(self.n_layer - 1):
            d = forward_per_layer(i, d)
        d = self['GRUAtten'](c, d)

        return d

    def reset_state(self):
        for i in range(self.n_layer - 1):
            self['GRU_{}'.format(i)].reset_state()
        self['GRUAtten'].reset_state()


class GRUAtten(Chain):
    """GRU unit with attention for decoder
    c: Encoder's output
    """

    def __init__(self, in_size, out_size, c_out_size):
        self.in_size = in_size
        self.d_c_dot_size = out_size  # https://github.com/sdlg/nlc/blob/master/nlc_model.py#L54

        super(GRUAtten, self).__init__(
            gru=L.StatefulGRU(in_size, out_size),
            linear_d=L.Linear(out_size, self.d_c_dot_size),
            linear_c=L.Linear(c_out_size, self.d_c_dot_size),
            linear_d_c_concat=L.Linear(c_out_size + out_size, out_size)
        )

    def __call__(self, c, d):  # (seq_size, batch_size, voc_size)
        """Spits out new sequence, d
        """

        self.c = c  # (c_seq_size, batch_size, c_out_size) c is encoder output
        seq_size = d.shape[0]

        d_new_list = []
        for t in range(seq_size):
            d_t_new = self.forward_t(d[t])  # (batch_size, hidden_size)
            d_new_list.append(d_t_new)

        d_new = F.concat([F.expand_dims(d_t, 0) for d_t in d_new_list], 0)  # (seq_size, batch_size, hidden_size)
        return d_new

    def forward_t(self, d_t):  # (batch_size, voc_size)
        """Feed-forward calculation for each output token d_t.
        """
        gru_out = self.gru(d_t)
        context_t = self._attend(gru_out)  # (batch_size, c_out_size)
        context_t_d_t_concat = F.concat([context_t, gru_out], axis=1)
        d_t_new = F.relu(self.linear_d_c_concat(context_t_d_t_concat))

        self.gru.h = d_t_new
        return d_t_new

    def reset_state(self):
        self.gru.reset_state()

    def _attend(self, gru_out):  # (batch_size, out_size)
        """Calculate context for each time step t.
        """
        c_seq_size, batch_size, c_out_size = self.c.shape

        # project d_t and c_t to the same space by linear transformation and tanh nonlineality
        phi_d_t = F.tanh(self.linear_d(gru_out))  # (batch_size, d_c_dot_size)
        phi_c_flatten = F.tanh(self.linear_c(F.reshape(self.c, (c_seq_size * batch_size, c_out_size))))  # (batch_size * c_seq_size, d_c_dot_size)

        # calculate weights(alpha) and context(weighted sum of c)
        phi_c = F.reshape(phi_c_flatten, (c_seq_size, batch_size, self.d_c_dot_size))
        phi_c_swapped = F.swapaxes(phi_c, 0, 1)  # (batch_size, c_seq_size, d_c_dot_size)
        phi_d_t_expanded = F.expand_dims(phi_d_t, axis=2)  # (batch_size, d_c_dot_size, 1)
        u_t_swapped = F.reshape(F.batch_matmul(phi_c_swapped, phi_d_t_expanded), (batch_size, c_seq_size))  # (batch_size, c_seq_size)
        u_t = F.swapaxes(u_t_swapped, 0, 1)  # (c_seq_size, batch_size)
        u_exp = F.exp(u_t)
        u_exp_sum = F.tile(F.expand_dims(F.sum(u_exp, axis=0), axis=0), (c_seq_size, 1))  # (c_seq_size, batch_size)
        alpha_t = u_exp / (u_exp_sum + 1e-06)  # (c_seq_size, batch_size)
        alpha_t_expanded = F.tile(F.expand_dims(alpha_t, 2), (1, 1, c_out_size))
        weighted_c = self.c * alpha_t_expanded  # (c_seq_size, batch_size, c_out_size)
        context_t = F.sum(weighted_c, axis=0)  # (batch_size, c_out_size)
        return context_t


class BidirectionalGRU(Chain):

    """Bidirectional GRU unit.
    """

    def __init__(self, in_size, hidden_size, out_size):
        """BidirectionalGRU constructor

        :param in_size: size of c (if c0, then VOC_SIZE)
        :param hidden_size: size of GRU hidden units
        :param output_size: size of output after linear transformation
        """
        super(BidirectionalGRU, self).__init__(
            f_gru=L.StatefulGRU(in_size, hidden_size),
            b_gru=L.StatefulGRU(in_size, hidden_size),
            pyr_linear=L.Linear(hidden_size * 2, out_size)  # concat h_{t} and h_{t+1}
        )

    def __call__(self, C):
        """Feedfoward calculation of c (if c is c0, c = X (input vector))

        :param C: array of c^{j-1}, whose size is (max_seq_size, batch_size, in_size(or VOC_SIZE))
        :return: array of c^{j}, whose size is (max_seq_size / 2, batch_size, hidden_size)
        """

        f_list = []
        for i in range(C.shape[0]):
            c = C[i]
            f = self.f_gru(c)
            f_list.append(f)

        b_list = []
        for i in reversed(range(C.shape[0])):
            c = C[i]
            b = self.b_gru(c)
            b_list.append(b)
        b_list.reverse()

        c_next_list = []
        t = 0
        # TODO: If the length of sequence is odd, tha last token is reused. Is it OK?
        while t < len(f_list):
            h_t = f_list[t] + b_list[t]  # (batch_size, hidden_size)
            if t != len(f_list) - 1:
                h_t_next = f_list[t + 1] + b_list[t + 1]  # (batch_size, hidden_size)
            else:
                h_t_next = h_t  # (batch_size, hidden_size)
            h = F.concat([h_t, h_t_next], 1)  # (batch_size, hidden_size * 2)
            c_next = self.pyr_linear(h)  # (batch_size, out_size)
            c_next_list.append(c_next)
            t += 2

        # TODO: There is no need to make C as 3-d array, make it list.
        C_next = F.concat([F.expand_dims(c_next, 0) for c_next in c_next_list], 0)  # (max_seq_len / 2, batch_size, out_size)

        return C_next

    def reset_state(self):
        self.f_gru.reset_state()
        self.b_gru.reset_state()
