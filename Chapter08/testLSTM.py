import tensorflow as tf
import reader
from tensorflow.python import debug as tf_debug
import numpy as np
import rnn_cell_impl
num_hidden=1
#https://statisticalinterference.wordpress.com/2017/06/01/lstms-in-even-more-excruciating-detail/
#https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9 
#input=tf.Variable(np.array([1,1,1,1,1]).reshape(5,1))
#input=tf.Variable([[1,1,1,1,1]],dtype=tf.float32)
input=tf.Variable([[1,2]],dtype=tf.float32,trainable=False)
input2=tf.Variable([[0.5,3]],dtype=tf.float32,trainable=False)
print(input)

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
with tf.Session() as sess:

    #cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=False)
    #[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    cell = rnn_cell_impl.BasicLSTMCell(num_hidden, state_is_tuple=True,forget_bias=0.15,reuse=False)

    #LSTMStateTuple(c=array([[ 0.,  0.,  0.,  0.,  0.]], dtype=float32), h=array([[ 0.,  0.,  0.,  0.,  0.]], dtype=float32))
    initial_state = cell.zero_state(1,input.dtype)

    (cell_out1, state) = cell(input, initial_state)
    #tf.get_variable_scope().reuse_variables()
    (cell_out, state) = cell(input2, state)

    #bp now
    #[cell_out, state]
    #[array([[ 0.77198118]], dtype=float32), LSTMStateTuple(c=array([[ 1.5176332]], dtype=float32), h=array([[ 0.77198118]], dtype=float32))]
    out_put1 = tf.Variable(0.5,dtype=tf.float32,trainable=False)
    out_put2 = tf.Variable(1.25,dtype=tf.float32,trainable=False)
    x_squared = tf.square(cell_out[0][0] -out_put2)+tf.square(cell_out1[0][0] - out_put1 )
    mse = x_squared / 2
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(mse)


    init_op = tf.global_variables_initializer()
    print("name:"+tf.get_variable_scope().name)
    sess.run(init_op)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
    #writer = tf.summary.FileWriter("E:/Program Files/Anaconda3/envs/tensorflow/Scripts/tensorlog", sess.graph)
    #writer.close()
    print(sess.run([cell_out, state]))
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    print(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    print(sess.run(train))
    print(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))