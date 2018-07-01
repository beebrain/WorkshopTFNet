
# coding: utf-8

# In[4]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# In[5]:


# Load example Data 
mnist = input_data.read_data_sets('MNIST-data', one_hot=True,reshape=False)


# In[6]:


#define  inital Weight and Bias function
def weightVariable(shapeW):
    init = tf.truncated_normal(shape=shapeW,stddev=0.1)
    return tf.Variable(initial_value=init)

def biasVariable(shapeB):
    init = tf.constant(0.5,shape=shapeB)
    return tf.Variable(initial_value=init)


# In[7]:


#define convolution Function 
def con2D(inputX,weightF,strite_num = 1):
    # tf.nn.conv2d( input, filter, strides, padding, use_cudnn_on_gpu=True,
    #               data_format='NHWC', dilations=[1, 1, 1, 1], name=None)
    # NHWC mean => [batch, height, width, channels]
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(input=inputX,filter=weightF,
                        strides=[1,strite_num,strite_num,1],padding="SAME")


# In[8]:


#define Maxpool Function
def maxPool(inputX,size=2,strite_num = 2):
    # tf.nn.max_pool(value,ksize,strides,padding,data_format='NHWC',name=None)
    return tf.nn.max_pool(value=inputX,ksize=[1,size,size,1],
                          strides=[1,strite_num,strite_num,1],padding="SAME")


# # Create Neural Network

# In[9]:


#create placeholder xs and ys , xs is an image input and ys is a label of class
#[optional]reshape for make sure the input is 28*28
tf.reset_default_graph()   # clear all graph

xs = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32,name="inputX")
ys = tf.placeholder(shape=[None,10],dtype=tf.float32,name="labelY")
xImage = tf.reshape(xs,[-1,28,28,1])


# In[10]:


#conv1
numFilter_1 = 32
filter_con1 = weightVariable([5,5,1,numFilter_1]) # patch 5x5, in size 1, out size 32
bias_con1 = biasVariable([numFilter_1])
feature_con1 = tf.nn.bias_add(con2D(inputX=xImage,weightF=filter_con1,strite_num=1),bias_con1)
feature_con1 = tf.nn.relu(feature_con1)   #relu Activation


# In[11]:


#Maxpool1
pool1 = maxPool(feature_con1)


# In[12]:


#conv2
numFilter_2 = 64
filter_con2 = weightVariable([5,5,numFilter_1,numFilter_2])
bias_con2 = biasVariable([numFilter_2])
feature_con2 = tf.nn.bias_add(con2D(inputX=pool1,weightF=filter_con2,strite_num=1),bias_con2)
feature_con2 = tf.nn.relu(feature_con2) #relu Activation


# In[13]:


#Maxpool2
pool2 = maxPool(inputX=feature_con2)


# In[14]:


#FullyConnected    #3136->512->512->10
layer_shape = pool2.get_shape()
TotalElement = layer_shape[1:4].num_elements()
print("ToTal Element = {:d}".format(TotalElement))


# In[15]:


FFconnect = tf.reshape(pool2,[-1,TotalElement])
wF1 = weightVariable(shapeW=[TotalElement,512])
bF1 = biasVariable(shapeB=[512])
FF_feature_1 = tf.matmul(FFconnect,wF1)+bF1
FF_feature_1 = tf.nn.relu(FF_feature_1)


# In[16]:


wF2 = weightVariable(shapeW=[512,10])
bF2 = biasVariable(shapeB=[10])
FF_feature_2 = tf.matmul(FF_feature_1,wF2)+bF2

prediction_out = tf.nn.softmax(FF_feature_2,name="Y_pre")


# define cross entropy
# $$H_{y'}(y) = - \sum_{i}  y'_i \times log(y_i)$$

# In[17]:


cost =tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction_out),reduction_indices=1),name="cost")
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)   


# In[18]:


#create accuracy tensor
y_predic_cls = tf.argmax(prediction_out,axis=1)
y_true_cls = tf.argmax(ys, axis=1)
correct_prediction = tf.equal(y_predic_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="ACC")


# In[19]:


# save Information
tf.summary.scalar('cross_entropy', cost)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


# In[22]:



with tf.Session() as sess:
    initValue = tf.global_variables_initializer()
    sess.run(initValue)
    train_summary_file = tf.summary.FileWriter("Log/train",graph=sess.graph)
    test_summary_file  = tf.summary.FileWriter("Log/test",graph=sess.graph)
    #training section
    
    acc_val = 0
    for i in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  #get data from mnist 
        _,summary_out  = sess.run([optimizer,summary_op], feed_dict={xs: batch_xs, ys: batch_ys})  #feed data to tensorflow  we have 2 branch optimizer and summary_op
        train_summary_file.add_summary(summary_out,i)  #save train_summary
        if i % 100 == 0:
            # success ?
            ta, tc = sess.run([accuracy, cost], feed_dict={xs: batch_xs, ys: batch_ys})    #feed train data to tensor fro accuracy and cost
            test_data = {xs: mnist.test.images, ys: mnist.test.labels}                             # get new data mnist for test
            va, vc, summary_test = sess.run([accuracy, cost, summary_op], feed_dict=test_data)                   #feed test data to tensor for acc and cost
            print("Step : %d Batch : acc = %.4f loss = %.4f | Test acc = %.4f loss = %.4f" % (i, ta, tc, va, vc))   #print imformation
            test_summary_file.add_summary(summary_test,i)
            if va > acc_val:
                
                # Save the variables to disk.
                save_path = saver.save(sess, "./tmp/bestACC")
                print("Model saved improve from: {:f}  to {:f}".format(acc_val,va))
                acc_val  = va


# # Restore Graph and weight

# In[94]:


# delete the current graph
tf.reset_default_graph()

# import the graph from the file
imported_graph = tf.train.import_meta_graph('./tmp/bestACC.meta')
# list all the tensors in the graph
with tf.Session() as sess:
    initValue = tf.global_variables_initializer()
    sess.run(initValue)
    imported_graph.restore(sess, './tmp/bestACC')
    test_data = {"inputX:0":mnist.test.images,"labelY:0":mnist.test.labels}
    a = sess.run(["ACC:0"],feed_dict=test_data)
    print("Step : %d Batch : Test acc = {:2.4f}".format(a[0]))   #print information


# In[117]:


with tf.Session() as sess:
    imported_graph.restore(sess, './tmp/bestACC')
    image = mnist.test.images[0]
    image = np.reshape(image,[28,28])
    plt.imshow(image)
    plt.show()
    
    
    image = np.reshape(image,[-1,28,28,1])
    a = sess.run(["Y_pre:0"],feed_dict= {"inputX:0":image})
    print(a)
    print(np.argmax(a[0],axis=1))

