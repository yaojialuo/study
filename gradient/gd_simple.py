import tensorflow as tf
from tensorflow.python import debug as tf_debug
x = tf.Variable(2, name='x', dtype=tf.float32)
z = tf.Variable(2, name='z', dtype=tf.float32)
Y = tf.placeholder("float")
#log_x = tf.log(x)
#log_x_squared = tf.square(log_x)
x_squared=tf.square(x+Y)
mse=x_squared/2
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(mse)
#y=tf.Variable(5.0, dtype=tf.float32)
init = tf.initialize_all_variables()

#yx=tf.placeholder(tf.float32)
def optimize():
    with tf.Session() as session:
        session.run(init)
        #session = tf_debug.LocalCLIDebugWrapperSession(sess=session)
        #session.run(train, feed_dict={Y: 6.0})
        #writer = tf.summary.FileWriter("E:/Program Files/Anaconda3/envs/tensorflow/Scripts/tensorlog", session.graph)
       # writer.close()


        # print(Y,y)
        # #session.run(x_squared,feed_dict={Y:1.6})
        # print("starting at", "x:", session.run([x,x_squared],feed_dict={Y:2.0}))
        for step in range(10):
             session.run([train],feed_dict={Y:6.0})
             print("x,z:", session.run([x, z,x_squared],feed_dict={Y:6.0}))


optimize()