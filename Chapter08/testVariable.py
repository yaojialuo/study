import tensorflow as tf



with tf.variable_scope("foo") as foo_scope:
    arr1 = tf.get_variable("arr1", shape=[2, 10], dtype=tf.float32)
    assert foo_scope.name == "foo"
with tf.variable_scope("bar"):
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name == "bar/baz"
        with tf.variable_scope(foo_scope) as foo_scope2:
            foo_scope2.reuse_variables()
            arr1 = tf.get_variable("arr1", shape=[2, 10], dtype=tf.float32)
            arr2 = tf.get_variable("arr1", shape=[2, 10], dtype=tf.float32)
            print(arr1)
            print(arr2)
            print(foo_scope2.name)
            assert foo_scope2.name == "foo"  # Not changed.

quit()
#http://blog.csdn.net/john_xyz/article/details/69053702
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

# def my_image_filter(input_images):
#     with tf.variable_scope("conv1"):
#         tf.get_variable_scope().reuse_variables()
#         # Variables created here will be named "conv1/weights", "conv1/biases".
#         relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
#     with tf.variable_scope("conv2"):
#         tf.get_variable_scope().reuse_variables()
#         # Variables created here will be named "conv2/weights", "conv2/biases".
#         return conv_relu(relu1, [5, 5, 32, 32], [32])
#
# image1=tf.get_variable("image1",[5, 5, 32, 32], initializer=tf.constant_initializer(0.0))
# image2=tf.get_variable("image2",[5, 5, 32, 32], initializer=tf.constant_initializer(0.0))
#
# result1 = my_image_filter(image1)
# result2 = my_image_filter(image2)


# with tf.variable_scope("image_filters") as scope:
#     result1 = my_image_filter(image1)
#     scope.reuse_variables()
#     result2 = my_image_filter(image2)
#quit()
#TF Boys (TensorFlow Boys ) 养成记（三）： TensorFlow 变量共享 http://www.cnblogs.com/Charles-Wan/p/6200446.html
#tensorflow学习笔记（二十三）：variable与get_variable http://blog.csdn.net/u012436149/article/details/53696970
with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    with tf.variable_scope("conv1") as inner:
        #tf.get_variable_scope().reuse_variables() can not appear here
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

