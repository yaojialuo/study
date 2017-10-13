import tensorflow as tf


with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
    tf.get_variable_scope().reuse_variables()
    var3_reuse = tf.get_variable(name='var3', )
    var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var3.name)  # a_variable_scope/var3:0
    print(sess.run(var3))  # [ 3.]
    print(var3_reuse.name)  # a_variable_scope/var3:0
    print(sess.run(var3_reuse))  # [ 3.]
    print(var4.name)  # a_variable_scope/var4:0
    print(sess.run(var4))  # [ 4.]
    print(var4_reuse.name)  # a_variable_scope/var4_1:0
    print(sess.run(var4_reuse))  # [ 4.]

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
print(v,v1)
assert v1 == v
quit()
with tf.name_scope("hello") as name_scope:
    arr1 = tf.get_variable("arr1", shape=[2,10],dtype=tf.float32)

    print(name_scope)
    print(arr1.name)
    print("scope_name:" , tf.get_variable_scope().original_name_scope)

print("========================================")
with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')
  print(a.name)
  print(W.name)
  print(b.name)

print("========================================")

