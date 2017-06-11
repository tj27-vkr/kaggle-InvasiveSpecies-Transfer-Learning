
# coding: utf-8

# In[1]:

import tensorflow as tf
import pandas as pd
import os
import numpy as np


# In[2]:

traindata = pd.read_csv("train_labels.csv")
traindata.head()


# Using the labels data, rename the training images, to categorize the data.

# In[16]:
# uncomment this block to rename and categorize the training images
# for i in range (0,len(traindata)):
#     image_class = traindata.ix[i][1]
#     os.rename("train/{}.jpg".format(i+1), "train/{}_{}.jpg".format(image_class,i+1))


# Train the images on InceptionV3 model using transfer learning.
# !python ../TensorFlow/tensorflow/tensorflow/examples/image_retraining/retrain.py \
# --bottleneck_dir=tf_files/bottlenecks \
# --how_many_training_steps 4000 \
# --model_dir=tf_files/inception \
# --output_graph=tf_files/retrained_graph.pb \
# --output_labels=tf_files/retrained_labels.txt \
# --summaries_dir=tf_files/logs \
# --image_dir=train
# 
# Once the training is complete, the following code predicts the test data using the trained and the saved model.

# Loading the trained model and the labels

# In[3]:

classes = [line.rstrip() for line in tf.gfile.GFile("tf_files/retrained_labels.txt")]
with tf.gfile.FastGFile("tf_files/retrained_graph.pb",'rb') as fd:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fd.read())
    _ = tf.import_graph_def(graph_def, name='')


# In[ ]:

predictions = []
#predictions.append(0)
cnt = 0
for i in range (1,len(os.listdir('test'))):
    input_image = tf.gfile.FastGFile('test/{}.jpg'.format(str(i)), 'rb').read()
    with tf.Session() as sess:
        final_layer_ouput = sess.graph.get_tensor_by_name("final_result:0")
        prediction = sess.run(final_layer_ouput, {'DecodeJpeg/contents:0':input_image})
        print(i)
        print(classes)
        print(prediction)
        predictions.append(prediction[0][1])


# In[ ]:

df = pd.DataFrame({"name":np.arange(len(predictions)), "invasive":predictions})
df.head()
df.to_csv('ouput.csv', index = 0)


# In[ ]:



