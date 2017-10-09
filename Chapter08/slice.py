import tensorflow as tf
data = [[[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]]
x = tf.strided_slice(data,[0,0,0],[1,1,1])
c=tf.Variable(5)
clip=tf.clip_by_value(c,6,20)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(clip))
    print(sess.run(x))


x = tf.slice(data,[0,0,0],[2,2,2])
with tf.Session() as sess:
    print(sess.run(x))