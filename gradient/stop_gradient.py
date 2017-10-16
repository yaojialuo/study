#http://blog.csdn.net/u012436149/article/details/53905797

import tensorflow as tf
def ex1():
    w1 = tf.Variable(2.0)
    w2 = tf.Variable(2.0)

    a = tf.multiply(w1, 3.0)
    a_stoped = tf.stop_gradient(a)

    # b=w1*3.0*w2
    b = tf.multiply(a, w2)
    gradients = tf.gradients(b, xs=[w2])
    print(gradients)
    opt = tf.train.GradientDescentOptimizer(0.1)
    train_op = opt.apply_gradients(zip(gradients, [w2]))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(gradients))
        print(sess.run([train_op,w2]))

def ex2():
    a = tf.Variable(1.0)
    b = tf.Variable(1.0)

    c = tf.add(a, b)

    c_stoped = tf.stop_gradient(c)

    d = tf.add(a, b)

    e = tf.add(c, d)

    gradients = tf.gradients(e, xs=[a, b])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(c_stoped))
        print(sess.run(gradients))


def ex3():
    w1 = tf.Variable(2.0,name="w1")
    w2 = tf.Variable(2.0,name="w2")
    a = tf.multiply(w1, 3.0,name="a")
    a_stoped = tf.stop_gradient(a)

    # b=w1*3.0*w2
    b = tf.multiply(a_stoped, w2,name="b")

    opt = tf.train.GradientDescentOptimizer(0.1)

    gradients = tf.gradients(b, xs=tf.trainable_variables())

    tf.summary.histogram("0", 0)  # 这里会报错，因为gradients[0]是None
    # 其它地方都会运行正常，无论是梯度的计算还是变量的更新。总觉着tensorflow这么设计有点不好，
    # 不如改成流过去的梯度为0
    train_op = opt.apply_gradients(zip(gradients, tf.trainable_variables()))

    print(gradients)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(tf.trainable_variables()))
        print(sess.run(train_op))
        print(sess.run([w1, w2]))


def ex4():
    import tensorflow as tf

    with tf.device('/cpu:0'):
        a = tf.constant(1.)
        b = tf.pow(a, 2)
        grad = tf.gradients(ys=b, xs=a)  # 一阶导
        print(grad[0])
        grad_2 = tf.gradients(ys=grad[0], xs=a)  # 二阶导
        grad_3 = tf.gradients(ys=grad_2[0], xs=a)  # 三阶导
        print(grad)
        print(grad_2)
        print(grad_3)

    with tf.Session() as sess:
        print(sess.run(grad))
        print(sess.run(grad_2))
        print(sess.run(grad_3))
ex1()
print("ex2===============================================================")
ex2()
print("ex3===============================================================")
ex3()
print("ex4===============================================================")
ex4()