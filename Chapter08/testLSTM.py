import tensorflow as tf
import reader
from tensorflow.python import debug as tf_debug
import numpy as np
num_hidden=5
#input=tf.Variable(np.array([1,1,1,1,1]).reshape(5,1))
input=tf.Variable([[1,1,1,1,1]],dtype=tf.float32)
print(input)


with tf.Session() as sess:

    #cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=False)
    #[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)

    #LSTMStateTuple(c=array([[ 0.,  0.,  0.,  0.,  0.]], dtype=float32), h=array([[ 0.,  0.,  0.,  0.,  0.]], dtype=float32))
    initial_state = cell.zero_state(1,input.dtype)

    (cell_out, state) = cell(input, initial_state)
    init_op = tf.initialize_all_variables()

    sess.run(init_op)
    print(sess.run([cell_out, state]))