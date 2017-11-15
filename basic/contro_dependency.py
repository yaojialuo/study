#https://github.com/vahidk/EffectiveTensorflow
import tensorflow as tf
a = tf.Variable(1)
b = tf.constant(2)
a+=1
with tf.control_dependencies([]):
    c = a + b



sess = tf.Session()
for i in range(10):
    sess.run(tf.global_variables_initializer())
    print(sess.run([a, c]))