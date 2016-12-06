import tensorflow as tf
from ptr_decoder import pointer_decoder
import numpy as np


class PointerNetwork(object):
    def __init__(self, hidden_unit, lr, grad_clip, data, name=None, max_seq_len=20, layer=1, batch_size=1, input_dim=2):
        """
        Args:
            hidden_unit: int
                number of hidden units or hidden size of LSTM cell
            lr: float32
                learning rate
            grad_clip: float32
                clip the gradient in [-grad_cli, grad_clip]
            data: a function n data_generator
                defines what data to generate
            name: str
                the name of the model
            max_seq_len: int
                the maximum sequence in one run
            layer: int
                number of layers in this pointer network
            batch_size: int
                how many samples to process in each process
            input_dim: int
                input dimension of pointer net
        """
        self.name = name if name is not None else "pointer_network"
        self.hidden_unit = hidden_unit
        self.max_seq_len = max_seq_len
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit, state_is_tuple=True)
        self.lr = lr
        self.hidden_unit = hidden_unit
        self.data = data

        # define encoder inputs
        self.encoder_inps = []
        for i in range(0, max_seq_len):
            encoder_inp = tf.placeholder(dtype=tf.float32, name="encoder_input_%s" % i, shape=(batch_size, input_dim))
            self.encoder_inps.append(encoder_inp)

        self.decoder_inps = []
        for i in range(0, max_seq_len):
            decoder_inp = tf.placeholder(dtype=tf.float32, name="decoder_input_%s" % i, shape=(batch_size, input_dim))
            self.decoder_inps.append(decoder_inp)

        self.targets = []
        for i in range(0, max_seq_len):
            target = tf.placeholder(dtype=tf.float32, name="target_%s" %i, shape=(batch_size, max_seq_len))
            self.targets.append(target)

        if layer > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell(cells=[self.cell] * layer)

        self.decoder_outputs, self.predictions = self.build()
        self.train_op, self.loss, self.test_loss, self.acc = self.build_optimizer()
        self.writer = tf.train.SummaryWriter("summary/%s" % data.__name__)
        self.writer.add_graph(tf.get_default_graph())
        self.summary = self.build_summary()

    def build(self):
        """
        Build the pointer network.
        The pointer network is consisted of 2 components:
            1.encoder: A regular rnn based on LSTM cell. outputs, final_state = rnn(cell ...)
            2.decoder: In training time, receives final_state and decoder_output(targets) and
                       produce the output to minimize the loss (cross-entropy)
                       In testing time, receives final_state and encoder_input(inputs) and
                       produce the output.
        Returns:
            If in test mode, return outputs from decoder.
            If in train mode, return outputs and losses.
        """
        with tf.variable_scope("pointer_net"):
            # encoder outputs need for attention
            encoder_outputs, final_state = tf.nn.rnn(self.cell, self.encoder_inps, dtype=tf.float32)

            # encoder_outputs = [tf.zeros(shape=(self.batch_size, self.hidden_unit), dtype=tf.float32)] + encoder_outputs
            encoder_outputs = [tf.reshape(out, (-1, 1, self.hidden_unit)) for out in encoder_outputs]
            encoder_outputs = tf.concat(1, encoder_outputs)

            # run decoder, for training to emulate http://arxiv.org/abs/1506.03099, and it is proved to have better
            # performance
            decoder_outputs = pointer_decoder(self.cell, self.decoder_inps, final_state, encoder_outputs,
                                              feed_prev=True, encoder_inputs=self.encoder_inps)
            tf.get_variable_scope().reuse_variables()
            predictions = pointer_decoder(self.cell, self.decoder_inps, final_state, encoder_outputs,
                                          encoder_inputs=self.encoder_inps, feed_prev=True)
            return decoder_outputs, predictions

    def build_summary(self):
        tf.summary.scalar("train_loss", self.loss)
        tf.summary.scalar("test_loss", self.test_loss)
        tf.summary.scalar("accuracy", self.acc)
        return tf.merge_all_summaries()

    def feed_dict(self, encoder_inpt_data, decoder_inpt_data, target_data):
        feed_dict = {}
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
        with tf.name_scope("train_loss"):
            loss = 0.0
            for output, target in zip(self.decoder_outputs, self.targets):
                # add loss over batch
                loss += tf.nn.softmax_cross_entropy_with_logits(output, target)
            loss = tf.reduce_mean(loss)

        with tf.name_scope("test_loss"):
            test_loss = 0.0
            for prediction, target in zip(self.predictions, self.targets):
                test_loss += tf.nn.softmax_cross_entropy_with_logits(prediction, target)
            test_loss = tf.reduce_mean(test_loss)

        with tf.name_scope("accuracy"):
            predictions = self.__split(tf.pack(self.predictions))
            targets = self.__split(tf.pack(self.targets))
            acc = 0.0
            for prediction, target in zip(predictions, targets):
                predict_index = tf.argmax(prediction, axis=1)
                target_index = tf.argmax(target, axis=1)
                acc += tf.to_float(tf.reduce_all(tf.equal(predict_index, target_index)))
            acc = tf.div(acc, self.batch_size)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.RMSPropOptimizer(self.lr)
            grads = optimizer.compute_gradients(loss, tf.trainable_variables())
            grads = [(tf.clip_by_value(grad, -self.grad_clip, self.grad_clip), var)
                     for grad, var in grads]
            train_op = optimizer.apply_gradients(grads)
        return train_op, loss, test_loss, acc

    def __split(self, x):
        """
        Split x into a list of Tensors with length x.dim[1],
        each Tensor has shape (x.dim[0], x.dim[2])

        Args:
            x: tensor to be split
        """
        batch_size, seq_len, dim = x.get_shape().as_list()
        y = tf.transpose(x, (1, 0, 2))
        y = tf.reshape(y, (-1, dim))
        y = tf.split(num_split=self.batch_size, split_dim=0, value=y)
        return y

    def train(self, sess, epoch=10000, print_every=100, test_every=100):
        """
        Train the model with given data generator

        Args:
            sess: tf.Session()
                the session of the graph
            epoch: int
                the maximum epoch to be trained
            print_every: int
                how many epochs to print the loss
            test_every: int
                how many epochs to print the test information
        """
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        for i in range(0, epoch):
            if i % test_every == 0:
                # test
                encoder_inpts, targets, decoder_inpts = self.data(self.max_seq_len, self.batch_size, is_train=False)
                feed_dict = self.feed_dict(encoder_inpts, decoder_inpts, targets)
                test_loss, acc, predictions, summary = sess.run([self.test_loss, self.acc, self.predictions,
                                                                 self.summary], feed_dict=feed_dict)
                # add summary
                self.writer.add_summary(summary, i)

                # sample
                sample_idx = np.random.randint(0, self.batch_size)
                prediction = np.argmax(np.transpose(predictions, (1, 0, 2))[sample_idx], axis=1)
                target = np.argmax(np.transpose(targets, (1, 0, 2))[sample_idx], axis=1)
                encoder_inpts = np.transpose(encoder_inpts, (1, 0, 2))
                target = encoder_inpts[sample_idx][target]
                prediction = encoder_inpts[sample_idx][prediction]
                if self.data.__name__ == "sort":
                    target = target.flatten()
                    prediction = prediction.flatten()
                else:
                    #target = encoder_inpts
                    pass

                print("------------------Testing-------------------")
                print("Epoch %s:" % i)
                print("Sample:\nground truth: %s\nprediction %s"
                      % (target, prediction))
                print("Test loss is %s" % test_loss)
                print("Test accuracy is %s" % acc)
                print("--------------------Done--------------------")
                # end of testing

            # train
            encoder_inpts, targets, decoder_inpts = self.data(self.max_seq_len, self.batch_size)
            feed_dict = self.feed_dict(encoder_inpts, decoder_inpts, targets)
            loss, summary, _ = sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)

            # add summary
            self.writer.add_summary(summary, i)
            if i % print_every == 0:
                print("Epoch %s: training loss is %s" % (i, loss))

    def save(self):
        pass

    def restore(self):
        pass
