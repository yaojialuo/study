import tensorflow as tf
import reader
from tensorflow.python import debug as tf_debug
import numpy as np
import rnn_cell_impl
num_hidden=1
#input=tf.Variable(np.array([1,1,1,1,1]).reshape(5,1))
#input=tf.Variable([[1,1,1,1,1]],dtype=tf.float32)
input=tf.Variable([[1,2]],dtype=tf.float32)
print(input)

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
with tf.Session() as sess:

    #cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=False)
    #[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    cell = rnn_cell_impl.BasicLSTMCell(num_hidden, state_is_tuple=True,forget_bias=0.15)

    #LSTMStateTuple(c=array([[ 0.,  0.,  0.,  0.,  0.]], dtype=float32), h=array([[ 0.,  0.,  0.,  0.,  0.]], dtype=float32))
    initial_state = cell.zero_state(1,input.dtype)

    (cell_out, state) = cell(input, initial_state)
    init_op = tf.global_variables_initializer()

    sess.run(init_op)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
    writer = tf.summary.FileWriter("E:/Program Files/Anaconda3/envs/tensorflow/Scripts/tensorlog", sess.graph)
    writer.close()
    print(sess.run([cell_out, state]))