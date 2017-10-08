import tensorflow as tf
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
print(y)
sess = tf.Session()

init_op = tf.global_variables_initializer()
sess.run(init_op)

#print(sess.run(y, feed_dict={x: [[0.7,0.9]]}))
print(sess.run(y, feed_dict={x: [[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
print(sess.run(tf.contrib.layers.l2_regularizer(.5)(y),feed_dict={x: [[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
quit()
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print(result)
sess = tf.InteractiveSession()
print(result.eval())
sess.close()

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", [1], initializer=tf.zeros_initializer())  # 设置初始值为0

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", [1], initializer=tf.ones_initializer())  # 设置初始值为1

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))