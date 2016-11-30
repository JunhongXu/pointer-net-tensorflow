import numpy as np


def sorting_generator(seq_len, batch_size, input_dim=1, is_train=True):
    """
    Generate sorting data for pointer networks

    Args:
        seq_len: int
            the maximum sequence length in generated data
        batch_size: int
            the batch size feed into the model
        input_dim: int
            the input dimension of the model
        is_train: bool
            whether to generate training data or not

    Returns: three elements
        1. encoder inputs: a random shuffled batch_size * sequence_len * input_dim numpy array (to be sorted)
        2. targets: a sorted batch_size * sequence_len * sequence_len numpy array.
        3. decoder inputs:
            If is_train is True, returns sorted encoder_iputs[:, 1:, :], and the first sequence should be all 0's
            for "GO" symbol.
            If is_train is False, returns encoder_inputs[:, 1:, :], and the first sequence should be all 0's for
            "GO" symbol.
    """
    # define "GO" symbol
    GO = np.ones((1, batch_size, input_dim))

    # generate random sequences
    sequences = np.random.random((seq_len, batch_size, input_dim))

    # sorted sequences
    sorted_sequences = np.sort(sequences, axis=0)
    sorted_index = np.argsort(sequences, axis=0).reshape((batch_size, seq_len * input_dim))
    # targets
    targets = np.zeros((seq_len+1, batch_size, seq_len + 1))
    for i in range(0, batch_size):
        targets[np.arange(seq_len), i, sorted_index[i]] = 1
    targets[-1, :, -1] = 1

    if is_train:
        decoder_inputs = np.append(GO, sorted_sequences[:, :, :], axis=0)
    else:
        decoder_inputs = np.append(GO, sequences[:, :, :], axis=0)

    return sequences, targets, decoder_inputs


def tsp_generator():
    pass
