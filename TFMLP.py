
# coding: utf-8

# In[13]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST-data', one_hot=True, reshape=True)


# In[14]:


tf.__version__


# In[15]:


#initial weight and bias
#WARNNING!!! Don't initial zero value into weight and bias
tf.reset_default_graph()   # clear all graph

W1 = tf.Variable(name="W1",initial_value=tf.truncated_normal(shape=[784,10],stddev=0.1),dtype=tf.float32)
b1 = tf.Variable(name="b1",initial_value=tf.constant(0.5,shape=[10]),dtype=tf.float32)


# In[16]:


#mnist Data is a digit image that have dimension is 28*28 pixels.
#create a model

Xdata = tf.placeholder(shape=[None,784],dtype=tf.float32)
Ydata = tf.placeholder(shape=[None,10],dtype=tf.float32)

#Hidden Layer
Layer1 = tf.matmul(Xdata,W1)+b1
#output layer
yout = tf.nn.softmax(Layer1,name="yout")


# In[17]:


# defined objective function and Optimizer 
cross_en = -tf.reduce_sum(tf.log(yout)*Ydata,reduction_indices=1)
cross_en = tf.reduce_mean(cross_en,name="cost_tensor")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(cross_en)


# In[18]:


#create evaluate tensor 
y_predict = tf.argmax(yout,axis=1)
y_true = tf.argmax(Ydata,axis=1)
correnctPredict = tf.equal(y_predict,y_true)
Acc = tf.reduce_mean(tf.cast(correnctPredict,dtype=tf.float32),name="acc_tensor")


# <img src="./imageFile/Tensor3.jpg">

# In[19]:


sess =  tf.Session()

# tensorboard write log
tf_train_board = tf.summary.FileWriter(logdir="./TFMLP/Train",graph=sess.graph)
tf_test_board = tf.summary.FileWriter(logdir="./TFMLP/Test",graph=sess.graph)

tf.summary.scalar("costValue",cross_en)
tf.summary.scalar("AccValue",Acc)
summary_op = tf.summary.merge_all()

#initial weight
init = tf.initializers.global_variables()
sess.run(init)


# In[ ]:


for i in range(60000): #100 epoach
    batchX,batchY = mnist.train.next_batch(100)
    summaryValue,_ = sess.run([summary_op,optimizer],feed_dict={Xdata:batchX,Ydata:batchY})
    tf_train_board.add_summary(summaryValue,global_step=i)
    if i%1000 == 0:
        (xTest, yTest) = (mnist.test.images,mnist.test.labels)
        vAC,vCro,summaryValue = sess.run([Acc,cross_en,summary_op],feed_dict={Xdata:xTest,Ydata:yTest})
        print("Loop {:d} Accuracy= {:f} Lost= {:f}".format(i,vAC,vCro))

