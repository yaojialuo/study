import tensorflow as tf
from tensorflow.python import debug as tf_debug
x = tf.Variable(1.0)
y = tf.add(x, x)

grad_debugger = tf_debug.GradientsDebugger()

debug_y = grad_debugger.identify_gradient(y)
z = tf.square(debug_y)

# Create a train op under the grad_debugger context.
with grad_debugger:
  train_op = tf.train.GradientDescentOptimizer(z)
  debug_x = grad_debugger.identify_gradient(x)
  x_grad = grad_debugger.gradient_tensor(x)
# Now we can reflect through grad_debugger to get the gradient tensor
# with respect to y.


#y_grad = grad_debugger.gradient_tensor(y)