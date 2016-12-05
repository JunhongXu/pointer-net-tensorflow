from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from pointer_network import PointerNetwork
import tensorflow as tf
import data_generator


flags = tf.app.flags


# Model
flags.DEFINE_float("lr", 1e-2, "Learning rate")
flags.DEFINE_integer("hidden_dim", 128, "Hidden dimension of LSTM")
flags.DEFINE_integer("layer", 1, "Layer size of LSTM")
flags.DEFINE_string("name", None, "Name of the model")
flags.DEFINE_integer("max_len", 10, "The length of the data")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_boolean("grad_clip", 10, "How much to clip the gradient")

# Misc
flags.DEFINE_integer("epochs", 10000, "Epoch")
flags.DEFINE_integer("print_every", 50, "Print training loss every ? epochs")
flags.DEFINE_integer("test_every", 100, "Test accuracy every ? epochs")
flags.DEFINE_string("task", "sort", "Task to be trained")
flags.DEFINE_bool("restore", True, "Whether to restore")
FLAGS = flags.FLAGS


def main(_):
    try:
        task = getattr(data_generator, FLAGS.task)
        input_dim = 2 if FLAGS.task is not "sort" else 1
    except ImportError:
        print("task %s has not been implemented" % FLAGS.task)
        raise
    sess = tf.Session()
    model = PointerNetwork(FLAGS.hidden_dim, FLAGS.lr, FLAGS.grad_clip, task, FLAGS.name, FLAGS.max_len, FLAGS.layer,
                           FLAGS.batch_size, input_dim)
    if FLAGS.restore:
        model.restore()

    model.train(sess, FLAGS.epochs, FLAGS.print_every, FLAGS.test_every)

if __name__ == '__main__':
    tf.app.run()

