#!/usr/bin/python

import tensorflow as tf
import numpy as np


class Model(object):

    image_size = 32
    num_labels = 11 # digits 0-9, + blank
    num_channels = 1 # grayscale

    batch_size = 64
    patch_size = 5

    depth1 = 16
    depth2 = 32
    depth3 = 64

    num_hidden1 = 64

    def __init__(self, size):
        self.graph = tf.Graph()
        with self.graph.as_default():

          # Input data.
          self.tf_test_dataset = tf.placeholder(tf.float32,
                                        shape=(size, self.image_size, self.image_size, self.num_channels))

          # Variables.
          layer1_weights = tf.get_variable("W1", shape=[self.patch_size, self.patch_size, self.num_channels, self.depth1],\
                   initializer=tf.contrib.layers.xavier_initializer_conv2d())
          layer1_biases = tf.Variable(tf.constant(1.0, shape=[self.depth1]), name='B1')
          layer2_weights = tf.get_variable("W2", shape=[self.patch_size, self.patch_size, self.depth1, self.depth2],\
                   initializer=tf.contrib.layers.xavier_initializer_conv2d())
          layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.depth2]), name='B2')
          layer3_weights = tf.get_variable("W3", shape=[self.patch_size, self.patch_size, self.depth2, self.num_hidden1],\
                   initializer=tf.contrib.layers.xavier_initializer_conv2d())
          layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden1]), name='B3')

          s1_w = tf.get_variable("WS1", shape=[self.num_hidden1, self.num_labels],\
                   initializer=tf.contrib.layers.xavier_initializer())
          s1_b = tf.Variable(tf.constant(1.0, shape=[self.num_labels]), name='BS1')
          s2_w = tf.get_variable("WS2", shape=[self.num_hidden1, self.num_labels],\
                   initializer=tf.contrib.layers.xavier_initializer())
          s2_b = tf.Variable(tf.constant(1.0, shape=[self.num_labels]), name='BS2')
          s3_w = tf.get_variable("WS3", shape=[self.num_hidden1, self.num_labels],\
                   initializer=tf.contrib.layers.xavier_initializer())
          s3_b = tf.Variable(tf.constant(1.0, shape=[self.num_labels]), name='BS3')
          s4_w = tf.get_variable("WS4", shape=[self.num_hidden1, self.num_labels],\
                   initializer=tf.contrib.layers.xavier_initializer())
          s4_b = tf.Variable(tf.constant(1.0, shape=[self.num_labels]), name='BS4')
          s5_w = tf.get_variable("WS5", shape=[self.num_hidden1, self.num_labels],\
                   initializer=tf.contrib.layers.xavier_initializer())
          s5_b = tf.Variable(tf.constant(1.0, shape=[self.num_labels]), name='BS5')

          # Model.
          def model(data, keep_prob, shape):
            conv = tf.nn.conv2d(data, layer1_weights, [1,1,1,1], 'VALID', name='C1')
            hidden = tf.nn.relu(conv + layer1_biases)
            lrn = tf.nn.local_response_normalization(hidden)
            sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S2')
            conv = tf.nn.conv2d(sub, layer2_weights, [1,1,1,1], padding='VALID', name='C3')
            hidden = tf.nn.relu(conv + layer2_biases)
            lrn = tf.nn.local_response_normalization(hidden)
            sub = tf.nn.max_pool(lrn, [1,2,2,1], [1,2,2,1], 'SAME', name='S4')
            conv = tf.nn.conv2d(sub, layer3_weights, [1,1,1,1], padding='VALID', name='C5')
            hidden = tf.nn.relu(conv + layer3_biases)
            hidden = tf.nn.dropout(hidden, keep_prob)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            #hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)
            logits1 = tf.matmul(reshape, s1_w) + s1_b
            logits2 = tf.matmul(reshape, s2_w) + s2_b
            logits3 = tf.matmul(reshape, s3_w) + s3_b
            logits4 = tf.matmul(reshape, s4_w) + s4_b
            logits5 = tf.matmul(reshape, s5_w) + s5_b
            return [logits1, logits2, logits3, logits4, logits5]

          # Training computation.
          [logits1, logits2, logits3, logits4, logits5] = model(self.tf_test_dataset, 1, [size, self.image_size, self.image_size, self.num_channels])

          predict = tf.stack([tf.nn.softmax(logits1),tf.nn.softmax(logits2),tf.nn.softmax(logits3),\
                                 tf.nn.softmax(logits4),tf.nn.softmax(logits5)])
                                 
          self.test_prediction = tf.transpose(tf.argmax(predict, 2))

          self.saver = tf.train.Saver()

    def predict(self, dataset):
        with tf.Session(graph=self.graph) as session:
            self.saver.restore(session, "SVHN_MODEL.ckpt")
            self.test_prediction = session.run(self.test_prediction, feed_dict={self.tf_test_dataset : dataset})
            return
