import tensorflow as tf
from ptr_decoder import pointer_decoder
import numpy as np


class PointerNetwork(object):
    def __init__(self, hidden_unit, lr, grad_clip, max_seq_len=20, layer=1, batch_size=1, input_dim=2):
        """
        Args:
        hidden_unit: int
            number of hidden units or hidden size of LSTM cell
        lr: float32
            learning rate
        grad_clip: float32
            clip the gradient in [-grad_cli, grad_clip]
        max_seq_len: int
            the maximum sequence in one run
        layer: int
            number of layers in this pointer network
        batch_size: int
            how many samples to process in each process
        input_dim: int
            input dimension of pointer net
        """
        self.hidden_unit = hidden_unit
        self.max_seq_len = max_seq_len
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit, state_is_tuple=True)
        self.lr = lr
        self.hidden_unit = hidden_unit

        # define ecoder inputs
        self.encoder_inps = []
        for i in range(0, max_seq_len):
            encoder_inp = tf.placeholder(dtype=tf.float32, name="encoder_input_%s" % i, shape=(batch_size, input_dim))
            self.encoder_inps.append(encoder_inp)

        self.decoder_inps = []
        for i in range(0, max_seq_len + 1):
            decoder_inp = tf.placeholder(dtype=tf.float32, name="decoder_input_%s" % i, shape=(batch_size, input_dim))
            self.decoder_inps.append(decoder_inp)

        self.targets = []
        for i in range(0, max_seq_len + 1):
            target = tf.placeholder(dtype=tf.float32, name="target_%s" %i, shape=(batch_size, max_seq_len + 1))
            self.targets.append(target)

        if layer > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells=[self.cell] * layer)

        self.decoder_outputs, self.predictions = self.build()
        self.train_op, self.loss, self.test_loss = self.build_optimizer()

    def build(self):
        """
        Build the pointer network.
        The pointer network is consisted of 2 components:
            1.encoder: A regular rnn based on LSTM cell. outputs, final_state = rnn(cell ...)
            2.decoder: In training time, receives final_state and decoder_output(targets) and
                       produce the output to minimize the loss (cross-entropy)
                       In testing time, receives final_state and encoder_input(inputs) and
                       produce the output.

        Args:
            feed_prev: If True, in test mode, else in train mode

        Returns:
            If in test mode, return outputs from decoder.
            If in train mode, return outputs and losses.
        """
        with tf.variable_scope("pointer_net"):
            # encoder outputs need for attention
            encoder_outputs, final_state = tf.nn.rnn(self.cell, self.encoder_inps, dtype=tf.float32)

            encoder_outputs = [tf.zeros(shape=(self.batch_size, self.hidden_unit), dtype=tf.float32)] + encoder_outputs
            encoder_outputs = [tf.reshape(out, (-1, 1, self.hidden_unit)) for out in encoder_outputs]
            encoder_outputs = tf.concat(1, encoder_outputs)

            # run decoder
            decoder_outputs = pointer_decoder(self.cell, self.decoder_inps, final_state, encoder_outputs)
            tf.get_variable_scope().reuse_variables()
            predictions = pointer_decoder(self.cell, self.decoder_inps, final_state, encoder_outputs,
                                          encoder_inputs=self.encoder_inps, feed_prev=True)
            return decoder_outputs, predictions

    def feed_dict(self, encoder_inpt_data, decoder_inpt_data, target_data):
        feed_dict={}
        for placeholder, data in zip(self.encoder_inps, encoder_inpt_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_inps, decoder_inpt_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.targets, target_data):
            feed_dict[placeholder] = data
        return feed_dict

    def build_optimizer(self):
        """
        Build the optimizer for training

        Args:
            lr: int
                Learning rate for the model

        Return:
            An optimizer op
        """
        loss = 0.0
        for output ,target in zip(self.decoder_outputs, self.targets):
            loss += tf.nn.sigmoid_cross_entropy_with_logits(output, target)
        loss = tf.reduce_mean(loss)

        test_loss = 0.0
        for prediction, target in zip(self.decoder_outputs, self.targets):
            test_loss += tf.nn.softmax_cross_entropy_with_logits(prediction, target)
        test_loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.minimize(loss)
        return train_op, loss, test_loss

