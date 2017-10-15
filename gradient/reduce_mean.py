import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable(1.0, tf.float32)
b = tf.Variable(1.0, tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
#loss = tf.reduce_mean(tf.square(linear_model - y)) # sum of the squares
loss = tf.square(linear_model - y)
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
# training data
# x_train = [2,4,6,8]
# y_train = [0,3,4,5]
x_train = [2,4]
y_train = [0,3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(500):
    sess.run(train, {x:x_train, y:y_train})

    # evaluate training accuracy
    curr_x,curr_W, curr_b, curr_loss  = sess.run([x,W, b, loss], {x:x_train, y:y_train})
    print("x: %s W: %s b: %s loss: %s"%(curr_x,curr_W, curr_b, curr_loss))