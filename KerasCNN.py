
# coding: utf-8

# In[1]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import os


# In[2]:


batch_size = 100
epochs = 1000


# In[3]:


def createModel(input_shape,num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),padding="same",activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model


# In[4]:



img_rows, img_cols = 28, 28
num_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#preprocess Reshape Xdata
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#preprocess encode one hot
y_train = keras.utils.np_utils.to_categorical(y_train,num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test,num_classes)

model = createModel(input_shape,num_classes)


# In[5]:


# save Graph Model
with open("./model.json","w") as jsonFile:
    model_json = model.to_json()
    jsonFile.write(model_json)
    print("written model.json ")


# In[6]:


#create callback_list
#The helper is to create tensorboard graph and checkpoint weight
weightPath = "./WeightFileKeras"
if not os.path.exists(weightPath):
    os.mkdir(weightPath)

weigth_file="%s/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"%(weightPath)

modelCheckpoint = keras.callbacks.ModelCheckpoint(weigth_file,monitor='val_loss',
                                  verbose=1,save_best_only=True,mode='min')

tbCallBack = keras.callbacks.TensorBoard(log_dir='./KerasLogs', histogram_freq=1, write_graph=True, write_images=True)
callbacks_list = [tbCallBack]


# In[ ]:


#loss function and optimizer are defined here.
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[ ]:


#start training!!
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks_list)

print("################## {:s} ####################".format("Finished Training"))


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {:f}'.format(score[0]))
print('Test accuracy: {:f}'.format(score[1]))


print("################## {:s} ####################".format("Finished Training"))
restoreModelFile = "model.json"


# In[ ]:


restoreWeightFile = "...."  #define weight here

with open(restoreModelFile, 'r') as jsonfile:
    print("Load model graph !!! ")
    loaded_model_json = jsonfile.read()
    jsonfile.close()
    model = model_from_json(loaded_model_json)
    print("Model graph is loaded!!")

    # load weights into new model
    model.load_weights(restoreWeightFile)
    print("Loaded model from disk")

