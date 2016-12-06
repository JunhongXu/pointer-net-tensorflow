# Pointer Networks Implementation in Tensorflow
Link to the paper: https://arxiv.org/pdf/1506.03134v1.pdf

Implemented Pointer Networks in Tensorflow. Two tasks implemented:
sorting and TSP(but this does not seem to converge, maybe due to not implement beam search.).

# Requirements
1. Python 2.7+. Anaconda environment is recommended: https://www.continuum.io/downloads
2. Tensorflow 0.12r, installation guid: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#requirements

# Usage
To train for sorting:

    $ python main.py --task=sort --max_len=10 --hidden_dim=128

To train for TSP:

    $ python main.py --task=tsp --max-len=4 --hidden_dim=512 --lr=1e-4

To visualize in Tensorboard:

    $ tensorboard --logdir=summary/tsp

    or

    $ tensorboard --logdir=summary/sort

# Issues:
Implement beam search in order to re-produce the result of the original paper

