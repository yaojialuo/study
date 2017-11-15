import tensorflow as tf
#
# a = tf.Variable(1)
#
# def body():
#     global a
#     for i in range(10):
#         a=a+1
#
#
# sess = tf.Session()
# for r in range(10):
#     body()
# sess.run(tf.global_variables_initializer())
# print(sess.run(a))


a = tf.get_variable("a", dtype=tf.int32, shape=[], initializer=tf.ones_initializer())
b = tf.constant(2)

f = tf.constant(6)

# Definition of condition and body
def cond(a, b, f):
    return a < f

def body(a, b, f):
    # do some stuff with a, b
    a = a + 1
    return a, b, f
# Loop, 返回的tensor while 循环后的 a，b，f
a, b, f = tf.while_loop(cond, body, [a, b, f])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    res = sess.run([a, b, f])
    print(res)













