import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result=tf.add_n([a,b],name="add")

#writer = tf.summary.FileWriter("tensorlog")
#writer = tf.summary.FileWriter("E:/Program Files/Anaconda3/envs/tensorflow/Scripts/tensorlog")
sess = tf.Session()

print(sess.run(result))
writer=tf.summary.FileWriter("E:/Program Files/Anaconda3/envs/tensorflow/Scripts/tensorlog",sess.graph)
writer.close()
print(tf.__version__)