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
        self.encoder_inps = tf.placeholder(dtype=tf.float32, name="encoder_inputs",
                                           shape=(batch_size, max_seq_len, input_dim))
        # targets should be one-hot vector
        self.targets = tf.placeholder(name="targets", shape=(batch_size, max_seq_len),
                                      dtype=tf.int32)
        self.decoder_inps = tf.placeholder(dtype=tf.float32, name="decoder_inputs",
                                           shape=(batch_size, max_seq_len, input_dim))
        if layer > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells=[self.cell] * layer)
        print "Building graph"
        # build training graph
        self.prediction, self.losses = self.build(feed_prev=False)
        # build testing graph, reusing variables
        tf.get_variable_scope().reuse_variables()
        self.validation = self.build(feed_prev=True)
        self.params = tf.trainable_variables()
        print "Building optimizer"
        self.opt = self.build_optimizer(lr)
        print "Finished"

    def build(self, feed_prev):
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
            # split placeholders into lists
            decoder_inps = self._split(self.decoder_inps)
            encoder_inps = self._split(self.encoder_inps)
            targets = tf.split(1, self.max_seq_len, self.targets)
            # targets = self._split(self.targets)
            initial_state = self.cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)

            # calculate encoder outputs and the final state, hids needed for attention
            hids, final_state = tf.nn.rnn(self.cell, encoder_inps, initial_state=initial_state)
            hids = tf.pack(hids, axis=1)

            if feed_prev:
                # for testing
                outputs = pointer_decoder(self.cell, decoder_inps, final_state, hids, feed_prev=feed_prev,
                                          encoder_inputs=encoder_inps)
                # now the outputs is seq*[batch*seq], convert it to batch*seq*seq
                # outputs = tf.pack(outputs, axis=1)
                return outputs

            # if for training, return loss and outputs
            outputs = pointer_decoder(self.cell, decoder_inps, final_state, hids)

            # outputs = tf.pack(outputs, axis=1)
            #  _outputs = tf.split(0, self.batch_size, outputs)
            # iterate through sequence to calculate cross-entropy loss

            losses = tf.nn.seq2seq.sequence_loss(outputs, targets,
                                                 [tf.constant(value=np.ones(self.batch_size),
                                                              dtype=tf.float32)] * self.max_seq_len,
                                                 average_across_timesteps=False)
            # for output, target in zip(_outputs, targets):
            #     loss = tf.nn.softmax_cross_entropy_with_logits(output, target)
            #     losses.append(loss)
            # losses = tf.transpose(tf.pack(losses), (1, 0))
            # losses = tf.reduce_mean(tf.reduce_sum(losses, reduction_indices=1))
        return outputs, losses

    def build_optimizer(self, lr):
        """
        Build the optimizer for training

        Args:
            lr: int
                Learning rate for the model

        Return:
            An optimizer op
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # clip the gradients
        grads = optimizer.compute_gradients(self.losses, var_list=self.params)
        clipped_grads = [(tf.clip_by_value(grad, clip_value_max=self.grad_clip, clip_value_min=-self.grad_clip), var)
                         for grad, var in grads]
        return optimizer.apply_gradients(clipped_grads)

    def _split(self, x):
        """
        Split x into a list of sequences

        Args:
            x: a 3D Tensor with shape (batch_size, max_seq_len, ?)
        Returns: a list of 2D Tensors with shape (batch_size, ?)
        """
        last_dim = x.get_shape()[-1].value
        x = tf.transpose(x, (1, 0, 2))
        x = tf.reshape(x, (-1, last_dim))
        x = tf.split(0, self.max_seq_len, x)
        return x
