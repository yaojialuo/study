import tensorflow as tf
data = [[[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]]
x = tf.strided_slice(data,[0,0,0],[1,1,1])
with tf.Session() as sess:
    print(sess.run(x))

x = tf.slice(data,[0,0,0],[2,2,2])
with tf.Session() as sess:
    print(sess.run(x))