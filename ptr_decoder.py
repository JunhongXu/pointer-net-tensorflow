import tensorflow as tf
from tensorflow.python.ops import rnn_cell


def pointer_decoder(cell, decoder_inputs, initial_state, attention_states,
                    scope=None, stddv=None, feed_prev=False, encoder_inputs=None):
    """
    Based on tensorflow implementation of decoder_with_attention. This computes the attention vector u without blending
    with encoder states as described in the paper.

    Args:
        cell: a tensorflow rnn_cell instance
        decoder_inputs: a list of 2D Tensors, each Tensor has shape (batch_size, input_size)
        initial_state: the final state after encoding, if cell is a LSTM,
            then the state should be a tuple of two tensors
        attention_states: a 3D Tensor with shape (batch_size, attention_len, attention_size). The usual case for
            attention_len should be the length of encoder hidden states. The usual case for attenton_size should be
            cell.state_size.
        scope: the name scope for this sub graph in tensorflow, if None, "decoder" will be used.
        stddv: the standard deviation for weight initialization, default is 1.0.
        feed_prev: whether feed the previous values to the decoder or not. If True, it should be in testing.
        encoder_inputs: should be provided in when feed_prev is True, a length of attention_len list of
            2D Tensor with shape (batch_size, cell.input_size)
    Returns:
        outputs: A list of 2D Tensors, each Tensor has shape of (batch_size, att_len).
            Each element in 2D Tensors represents the probability to choose to that input point.
            The outputs are computed as follows:
            1. compute the output of rnn at time t:
                out_t, state_t = cell(x_t, state_t-1)
            2. compute the attention mask at time t:
                att_t = V^t * tanh(W_att * att_state + W_out * out_t) (shape: (batch_size, attention_len))
            3. compute softmax distribution over time t:
                output_t = tf.nn.softmax(att_t) (shape: (batch_size, attention_len))
            At testing time, the input to next time step will be encoder_inputs[arg_max(output_t)]
        state: A tuple of 2D Tensors
    """
    # check inputs
    if not decoder_inputs:
        raise ValueError("Must provide inputs to pointer decoder")

    # check attention states
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of the attention states must be known")

    # check test time and training time inputs
    if feed_prev:
        if not encoder_inputs:
            raise ValueError("Encoder inputs should not be none in test time")
    else:
        if encoder_inputs:
            raise ValueError("Encoder inputs should not be provided in train time")

    with tf.variable_scope("decoder" or scope):
        attention_len = attention_states.get_shape()[1].value
        attention_size = attention_states.get_shape()[2].value
        state = initial_state
        prev = None
        outputs = []
        if attention_len is None:
            attention_len = attention_states.get_shape()[1]
        if encoder_inputs:
            encoder_inputs = tf.pack(encoder_inputs)
            seq_len, batch_size, dim = encoder_inputs.get_shape().as_list()
            y = tf.transpose(encoder_inputs, (1, 0, 2))
            y = tf.reshape(y, (-1, dim))
            encoder_inputs = tf.split(num_split=batch_size, split_dim=0, value=y)

        # reshape attention_state to 4D Tensor to do convolution operation
        attention_states = tf.reshape(attention_states, (-1, attention_len, 1, attention_size))

        # one-by-one convolution kernel
        kernel = tf.get_variable(name="Att_W1", shape=(1, 1, attention_size, attention_size), dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=stddv or 1.0))
        hidden_feature = tf.nn.conv2d(attention_states, kernel, (1, 1, 1, 1), padding="SAME")
        v = tf.get_variable(name="Att_V", shape=attention_size,
                            initializer=tf.truncated_normal_initializer(stddev=stddv or 1.0))

        # define attention function
        def attention(query):
            with tf.variable_scope("Attention"):
                # attention on query (decoder states)
                query_feature = rnn_cell._linear(query, bias=True, output_size=attention_size, scope="Att_W2")

                # reshape query_feature feature to (-1, 1, 1, attention_size) in order to do summation
                query_feature = tf.reshape(query_feature, (-1, 1, 1, attention_size))

                # compute attention vector u, should be (batch_size, attention_len)
                s = tf.reduce_sum(v * tf.nn.tanh(query_feature + hidden_feature), reduction_indices=[2, 3])
            return s

        # iterate over decoder inputs
        for index, inp in enumerate(decoder_inputs):
            if index > 0:
                tf.get_variable_scope().reuse_variables()

            if feed_prev and index > 0:  # testing
                prev = tf.arg_max(prev, dimension=1)
                indices = tf.range(0, batch_size)
                prev = tf.cast(prev, tf.int32)
                prev = tf.pack((indices, prev), 1)
                inp = tf.gather_nd(encoder_inputs, indices=prev)
            x = rnn_cell._linear(inp, cell.output_size, True, scope="embedding")
            # run one step of decoder
            hid, state = cell(x, state)
            # run attention
            output = attention(hid)
            prev = output
            outputs.append(output)
        return outputs



