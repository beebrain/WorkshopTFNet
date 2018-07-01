import time as t
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# Creates a graph.
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
	
	
print("CPU TEST")
ticks = t.time()
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session()
# Runs the op.
print(sess.run(c))
print("################# CPU Finished Time %f ##################"%(t.time()-ticks))


c = []
deviceName = get_available_gpus()
for d in deviceName:
    ticks = t.time()
    with tf.device(d):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
        c.append(tf.matmul(a, b))
        sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
        print(sess.run(sum))
        print("################# GPU Finished Time %f ##################"%(t.time()-ticks))