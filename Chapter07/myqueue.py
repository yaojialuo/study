import tensorflow as tf
import numpy as np
import threading
import time


queue = tf.FIFOQueue(100,"float")
enqueue_op = queue.enqueue([tf.random_normal([1])])
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
tf.train.add_queue_runner(qr)
out_tensor = queue.dequeue()
coord = tf.train.Coordinator()

with tf.Session() as sess:

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(3): print(sess.run(out_tensor))
    coord.request_stop()
    coord.join(threads)



#print([1]*5)
# def MyLoop(coord, worker_id):
#     while not coord.should_stop():
#         if np.random.rand()<0.1:
#             print("Stoping from id: %d\n" % worker_id)
#             coord.request_stop()
#         else:
#             print("Working on id: %d\n" % worker_id)
#         time.sleep(1)
#
# coord = tf.train.Coordinator()
# threads = [threading.Thread(target=MyLoop, args=(coord, i,)) for i in range(5)]
# for t in threads: t.start()
# coord.join(threads)



# q = tf.FIFOQueue(2, "int32")
# init = q.enqueue_many(([0, 10],))
# x = q.dequeue()
# y = x + 1
# q_inc = q.enqueue([y])
# with tf.Session() as sess:
#     init.run()
#     for _ in range(5):
#         #v, _ = sess.run([x, q_inc])
#         print(sess.run(y)) #1 11 then waiting for enqueue
#         #print(v)