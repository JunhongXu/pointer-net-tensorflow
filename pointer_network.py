import tensorflow as tf


class PointerNetwork(object):
    def __init__(self, hidden_unit, lr, grad_clip, layer=1, batch_size=1, input_dim=2):
        """
        Args:
        hidden_unit: int
            number of hidden units or hidden size of LSTM cell
        lr: float32
            learning rate
        grad_clip: float32
            clip the gradient in [-grad_cli, grad_clip]
        layer: int
            number of layers in this pointer network
        batch_size: int
            how many samples to process in each process
        input_dim: int
            how
        """
        self.hidden_unit = hidden_unit
        self.lr = lr
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.input_dim = input_dim

    def build(self):
        """
        Build the pointer network.
        The pointer network is consisted of 2 components:
            1.encoder: A regular rnn based on LSTM cell. outputs, final_state = rnn(cell ...)
            2.decoder: In training time, receives final_state and decoder_output(targets) and
                       produce the output to minimize the loss (cross-entropy)
                       In testing time, receives final_state and encoder_input(inputs) and
                       produce the output.
        """


