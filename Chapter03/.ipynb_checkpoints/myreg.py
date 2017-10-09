import tensorflow as tf

def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
    return var


# weights = tf.constant([[1.0,-2.0],[-3.0,4.0]])
# with tf.Session() as sess:
#     print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))
#     print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
