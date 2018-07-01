
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np

#defind weight
tf.reset_default_graph()
tf.set_random_seed(0)
xin = tf.placeholder(tf.float32,shape=[None,10])
w = tf.get_variable(name= "Weight1",initializer=tf.truncated_normal([10,3],stddev=0.5))
bias = tf.get_variable(name="bias1",initializer=tf.constant(0.5,shape=[3]))
y_out = tf.matmul(xin,w)+bias
yout = tf.nn.softmax(y_out)



# In[ ]:


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    xdata = np.array([1,2,3,4,5,6,7,8,9,10])
    xdata = xdata.reshape(-1,10)
    print(sess.run(w))
    print(sess.run(bias))
    print(sess.run(yout,feed_dict={xin:xdata}))
    print(sess.run(y_out,feed_dict={xin:xdata}))

