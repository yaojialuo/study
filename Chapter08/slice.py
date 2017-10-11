import tensorflow as tf

sig=tf.Variable([[-0.03335544,  0.09980205,  0.11589614,  0.08302781,  0.26093069]])
tan=tf.Variable([[ 0.33293939,  0.33492783,  0.73586965,  0.53402126,  0.69271994]])
sigmoid_i=tf.Variable([[-0.69492048, -0.68598044,  1.02461028,  0.13629574,  0.81286621]])
mul=sig*tan
sigmoid=tf.sigmoid(sigmoid_i)
data = [[[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]]
x = tf.strided_slice(data,[0,0,0],[1,1,1])
c=tf.Variable(5)
clip=tf.clip_by_value(c,6,20)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("sigmoid:",sess.run(sigmoid))
    print(sess.run(mul))
    print(sess.run(clip))
    print(sess.run(x))


x = tf.slice(data,[0,0,0],[2,2,2])
with tf.Session() as sess:
    print(sess.run(x))