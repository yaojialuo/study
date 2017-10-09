import tensorflow as tf
import reader
from tensorflow.python import debug as tf_debug
import numpy

DATA_PATH = "E:/study/ML/tensorflow/study/Chapter08/data"
train_data, valid_data, test_data, word_to_id = reader.ptb_raw_data(DATA_PATH)
id_to_word={value:key for key, value in word_to_id.items()}
print(len(train_data))
print(train_data[:100])
print(id_to_word.values())
#train_data=tf.constant(list(range(100)))
v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
result = reader.ptb_producer(train_data, 4, 5)

saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
with tf.Session() as sess:
    debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
    #writer = tf.summary.FileWriter("E:/Program Files/Anaconda3/envs/tensorflow/Scripts/tensorlog", sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(1):

        x, y = sess.run(result)
        #x, y = sess.run(result)
        #print(type(x.shape))

        nx = numpy.array([id_to_word[_] for _ in x.flatten()])
        nx = nx.reshape(x.shape);
        print(nx,x.shape)
        print(nx.dtype)
        #print("Y: ", y)
        #print(type(y))
        #print(sess.run(train_data))
        #saver_path = saver.save(sess, "model.ckpt")
    coord.request_stop()
    coord.join(threads)
    print(len(threads))
    #writer.close()



    # test
    # x = tf.Variable(1.0,name='x')
    # z=tf.identity(x,name='z')
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
    #     debug_sess.run(z)
    # exit()
    # test

    # identity

    # x = tf.Variable(1.0,name='x')
    # x_plus_1 = tf.assign_add(x, 1)
    #
    # with tf.control_dependencies([x_plus_1]):
    #     y = x
    #     z=tf.identity(x,name='z')
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
    #     #debug_sess(init)
    #     for i in range(5):
    #         print(sess.run([x_plus_1,y]))
    #         print(x is x_plus_1 )
    # print(debug_sess(y), debug_sess(x))



    # exit()

    # identity










    # def fab(max):
#     n, a, b = 0, 0, 1
#     while n < max:
#         yield b
#         # print b
#         a, b = b, a + b
#         n = n + 1
# for n in fab(5):
#     print(n)
#
# with tf.name_scope("hello") as name_scope:
#     arr1 = tf.get_variable("arr1", shape=[2,10],dtype=tf.float32)
#
#     print(name_scope)
#     print(arr1.name)
#     print("scope_name:%s " % tf.get_variable_scope().original_name_scope)
#
# quit()