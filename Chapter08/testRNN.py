#https://stackoverflow.com/questions/38994037/tensorflow-while-loop-for-training
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
#rnn_implementation
input=tf.Variable([[1]],dtype=tf.float32,trainable=False)
step=2
inputdata=tf.Variable([[[1],[1]]],dtype=tf.float32,trainable=False)
np.ones((1,2,1))
#[batch_size, max_time, ...]
inputdata=tf.Variable(np.ones((1,2,1)),dtype=tf.float32,trainable=False)
#<tf.Variable 'Variable_1:0' shape=(1, 2, 1) dtype=float32_ref>
print(inputdata)
output=tf.Variable([[step]],dtype=tf.float32,trainable=False)
print(input)
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

count=tf.Variable([[0]],dtype=tf.float32,trainable=False)
cell = rnn_cell_impl.BasicRNNCell(num_hidden)
initial_state = cell.zero_state(1, tf.float32)
print(initial_state)
mystate = cell.zero_state(1, tf.float32)


def cond(i,mystate):
    return i < 1


def body(i,mystate):
    # LSTMStateTuple(c=array([[ 0.,  0.,  0.,  0.,  0.]], dtype=float32), h=array([[ 0.,  0.,  0.,  0.,  0.]], dtype=float32))


    #tf.add_to_collection("state", state)
    #https: // github.com / tensorflow / tensorflow / issues / 4094  # issue-173787623
    #Any op created in a branch of a TensorFlow conditional or the body of a TensorFlow loop is marked as "non-fetchable", to prevent various programming errors.

    # _, state = cell(input, initial_state)
    # for r in range(step):
    #     if (r > 0):
    #         _, state = cell(input, state)


    _, state = tf.nn.dynamic_rnn(cell, inputdata,initial_state=initial_state, dtype=tf.float32)

    # tf.get_variable_scope().reuse_variables()
    train = tf.train.GradientDescentOptimizer(1).minimize(tf.square(state - output))
    with tf.control_dependencies([train]):
        #tf.assign(mystate,state)
        print(i)
        return i + 1,state


loop = tf.while_loop(cond, body, [tf.constant(0),mystate])
init_op = tf.global_variables_initializer()
print("name:" + tf.get_variable_scope().name)
with tf.Session() as sess:
    # cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=False)
    # [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]

    sess.run(init_op)
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
    #writer = tf.summary.FileWriter("E:/Program Files/Anaconda3/envs/tensorflow/Scripts/tensorlog", sess.graph)
    #writer.close()
    #print(sess.run(initial_state))

    print(sess.run([tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))
    print(sess.run(loop))
    print(sess.run([tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))

#1
# [[array([[-1.5],
#          [2.]], dtype=float32)]]
# (1, array([[-4.5]], dtype=float32))
# [[array([[37.5],
#          [-17.5]], dtype=float32)]]
#2
# [[array([[-1.5],
#        [ 2. ]], dtype=float32)]]
# (2, array([[-618.75]], dtype=float32))
# [[array([[-20447.25],
#        [ 46538.75]], dtype=float32)]]