import numpy as np
import itertools
import pandas


def sort(seq_len, batch_size, is_train=True):
    """
    Generate sorting data for pointer networks

    Args:
        seq_len: int
            the maximum sequence length in generated data
        batch_size: int
            the batch size feed into the model
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
    GO = np.ones((1, batch_size, 1))

    # generate random sequences
    sequences = np.random.random((seq_len, batch_size, 1))

    # sorted sequences
    sorted_sequences = np.sort(sequences, axis=0)
    sorted_index = np.squeeze(np.argsort(sequences, axis=0))
    # targets
    targets = np.zeros((seq_len, batch_size, seq_len))
    for i in range(0, batch_size):
        targets[np.arange(seq_len), i, sorted_index[:, i]] = 1
    # targets[-1, :, -1] = 1

    if is_train:
        decoder_inputs = np.append(GO, sorted_sequences[:-1, :, :], axis=0)
    else:
        decoder_inputs = np.append(GO, np.zeros((seq_len - 1, batch_size, 1)), axis=0)

    return sequences, targets, decoder_inputs


def tsp_a1():
    pass


def tsp_a2():
    pass


def tsp_a3():
    pass


def tsp_opt(points):
    """
    Calculate tsp optimal solution using dynamic programming
    # https://gist.github.com/mlalevic/6222750

    Args:
        points: numpy array
            Shape(seq_len, 2). Representing x, y coordinates of different cities
    Return:
        numpy array representing the order to visit
    """
    def length(x_coord, y_coord):
        return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))
    # calc all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for idx, dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
            for j in S - {0}:
                # this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                B[(S, j)] = min([(A[(S - {j}, k)][0] + all_distances[k][j], A[(S - {j}, k)][1] + [j])
                                 for k in S if k != 0 and k != j])
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    rres = res[1]
    # rres.append(0)
    rres = np.asarray(rres)
    return rres


def generate_tsp(data_size, seq_len):
    result = []
    nodes_list = []
    for i in range(0, data_size):
        print("Generating %sth data" % i)
        nodes = np.random.random(size=(seq_len, 2))
        nodes_list.append(nodes.flatten())
        result.append(tsp_opt(nodes))

    nodes_list = pandas.DataFrame(nodes_list)
    result = pandas.DataFrame(result)
    nodes_list.to_csv("data/tsp_seq20_opt_data.csv")
    result.to_csv("data/tsp_seq20_opt_solution.csv")


def tsp(seq_len, batch_size, is_train=True):
    """
    Generates travelling salesman data based on sequence length and batch size.
    The data are randomly chose from /data folder. The
    :param seq_len:
    :param batch_size:
    :param is_train:
    :return:
    """
    map()
    pass
