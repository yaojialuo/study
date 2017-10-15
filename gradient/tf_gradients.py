# right
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
w1 = tf.Variable([[1,2]])
#w1 = tf.placeholder(tf.int32)
w2 = tf.Variable([[3,4]])

res = tf.matmul(w1, [[2],[1]])

grads = tf.gradients(res,[w1],grad_ys=[tf.convert_to_tensor([[4]])])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(res))
    re = sess.run(grads)
    print(re)
#  [array([[2, 1]], dtype=int32)]