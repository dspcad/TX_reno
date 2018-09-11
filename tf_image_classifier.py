#!/usr/bin/python

import numpy as np
import csv
import tensorflow as tf
import time
from PIL import Image
from skimage import io
from skimage import transform
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
#import matplotlib.pyplot as plt

def drawBBox(img, pred_coordinates, ground_truth_coordinates):
  xmin = int(clipWidth(pred_coordinates[0])) 
  xmax = int(clipWidth(pred_coordinates[1])) 
  ymin = int(clipHeight(pred_coordinates[2])) 
  ymax = int(clipHeight(pred_coordinates[3])) 

  #print "image shape: ", img.shape
  #print "coordinate: ", coordinates

  for i in range(0,xmax-xmin):
    img[ymin][xmin+i][0] = 255
    img[ymin][xmin+i][1] = 0
    img[ymin][xmin+i][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmin][0] = 255
    img[ymin+i][xmin][1] = 0
    img[ymin+i][xmin][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmax][0] = 255
    img[ymin+i][xmax][1] = 0
    img[ymin+i][xmax][2] = 0

  for i in range(0,xmax-xmin):
    img[ymax][xmin+i][0] = 255
    img[ymax][xmin+i][1] = 0
    img[ymax][xmin+i][2] = 0

  xmin = int(ground_truth_coordinates[0]) - 1
  xmax = int(ground_truth_coordinates[1]) - 1
  ymin = int(ground_truth_coordinates[2]) - 1
  ymax = int(ground_truth_coordinates[3]) - 1

  #print "image shape: ", img.shape
  #print "coordinate: ", coordinates

  for i in range(0,xmax-xmin):
    img[ymin][xmin+i][0] = 0
    img[ymin][xmin+i][1] = 128
    img[ymin][xmin+i][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmin][0] = 0
    img[ymin+i][xmin][1] = 128
    img[ymin+i][xmin][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmax][0] = 0
    img[ymin+i][xmax][1] = 128
    img[ymin+i][xmax][2] = 0

  for i in range(0,xmax-xmin):
    img[ymax][xmin+i][0] = 0
    img[ymax][xmin+i][1] = 128
    img[ymax][xmin+i][2] = 0


  return img


def clipWidth(val):
  if val > 1280:
    return 1280

  if val < 0:
    return 0

  return val

def clipHeight(val):
  if val > 720:
    return 720

  if val < 0:
    return 0

  return val


def checkIOU(label_BBox, pred_BBox):
  #print "label_BBox shape: ", label_BBox.shape
  #print "pred_BBox shape: ", pred_BBox.shape

  IOU = np.zeros(label_BBox.shape[0])

  for i in range(label_BBox.shape[0]):
    ###############################
    #  check validity of pred box #
    #   (xmin, xmax, ymin, ymax)  #
    ###############################
    if pred_BBox[i][0] >= pred_BBox[i][1] or pred_BBox[i][2] >= pred_BBox[i][3]:
      IOU[i] = 0
    #elif label_BBox[i][0] == -1:
    #  if pred_BBox[i][0] < 0 or pred_BBox[i][1] <0 or pred_BBox[i][2] < 0 or pred_BBox[i][3] < 0:
    #    IOU[i] = 1
    #  else:
    #    IOU[i] = 0
    else:
      if checkIntersection(label_BBox[i], pred_BBox[i]) == 1:

        #xmin_A = limitWithinOne(pred_BBox[i][0])
        #xmax_A = limitWithinOne(pred_BBox[i][1])
        #ymin_A = limitWithinOne(pred_BBox[i][2])
        #ymax_A = limitWithinOne(pred_BBox[i][3])

        xmin_A = clipWidth(pred_BBox[i][0])
        xmax_A = clipWidth(pred_BBox[i][1])
        ymin_A = clipHeight(pred_BBox[i][2])
        ymax_A = clipHeight(pred_BBox[i][3])


        xmin_B = label_BBox[i][0]
        xmax_B = label_BBox[i][1]
        ymin_B = label_BBox[i][2]
        ymax_B = label_BBox[i][3]

        #print "pred BBox[0]: ", xmin_A
        #print "pred BBox[1]: ", xmax_A
        #print "pred BBox[2]: ", ymin_A
        #print "pred BBox[3]: ", ymax_A
        #print "width A: ", (xmax_A-xmin_A)
        #print "height A: ", (ymax_A-ymin_A)

        #print "label BBox[0]: ", xmin_B
        #print "label BBox[1]: ", xmax_B
        #print "label BBox[2]: ", ymin_B
        #print "label BBox[3]: ", ymax_B
        #print "width B: ", (xmax_B-xmin_B)
        #print "height B: ", (ymax_B-ymin_B)


        xmin_intersection = np.maximum(xmin_A, xmin_B)
        xmax_intersection = np.minimum(xmax_A, xmax_B)
        ymin_intersection = np.maximum(ymin_A, ymin_B)
        ymax_intersection = np.minimum(ymax_A, ymax_B)

        intersection_width  = xmax_intersection-xmin_intersection
        intersection_height = ymax_intersection-ymin_intersection

        if intersection_width < 0 or intersection_height < 0:
          IOU[i] = 0
        else:
          intersection_area = intersection_width*intersection_height
          area_two_boxes = (xmax_A-xmin_A)*(ymax_A-ymin_A) + (xmax_B-xmin_B)*(ymax_B-ymin_B)
          IOU[i] = intersection_area/(area_two_boxes-intersection_area) 

        #print "intersection_area: ", intersection_area
        #print "area_two_boxes: ", area_two_boxes
        #print "IOU[%d]: %f" % (i, IOU[i])
      else:
        IOU[i] = 0
      

  #print "xmin_union: ", xmin_union
  #print "ymin_union: ", ymin_union
  #print "xmax_union: ", xmax_union
  #print "ymax_union: ", ymax_union


    
  #print IOU
  return IOU

def checkIntersection(BBoxA, BBoxB):
  ###############################
  #       shape[0]: height      #
  #       shape[1]: width       #
  ###############################

  xmin = BBoxB[0] 
  xmax = BBoxB[1]
  ymin = BBoxB[2]
  ymax = BBoxB[3]


  #########################################
  #     BBox intersects the grid cells    #
  #########################################
  target_x = BBoxA[0] #xmin
  target_y = BBoxA[2] #ymin
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = BBoxA[0] #xmin
  target_y = BBoxA[3] #ymax
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = BBoxA[1] #xmax
  target_y = BBoxA[3] #ymax
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = BBoxA[1] #xmax
  target_y = BBoxA[2] #ymin
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1
 
 
  #########################################
  #       BBoxB is within in BBoxA        #
  #########################################
  if xmin >= BBoxA[0] and ymin >= BBoxA[2] and xmax <= BBoxA[1] and ymax <= BBoxA[3]:
    return 1


  return 0



if __name__ == '__main__':
  print '===== Start loading the labels of DAC Tracking datasets ====='
  infile = open("label.txt", "r")
  lines = infile.readlines()

  lines = map(lambda s: s.strip(), lines)
  label_dict = {}
  for l in lines:
    elements = l.split()
    label_dict[elements[0]] = elements[1]




  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 128

  K = 2 # number of classes
  G = 256 # number of grid cells
  P = 4  # four parameters of the bounding boxes
  lamda = 0.001

  NUM_FILTER_1 = 16
  NUM_FILTER_2 = 16
  NUM_FILTER_3 = 32
  NUM_FILTER_4 = 32
  NUM_FILTER_5 = 64
  NUM_FILTER_6 = 64

  NUM_NEURON_1 = 512
  NUM_NEURON_2 = 512

  DROPOUT_PROB = float(sys.argv[1])
  LEARNING_RATE = float(sys.argv[2])


  print 'Settings: '
  print '    Dropout: ', DROPOUT_PROB
  print '    Learning Rate: : ', LEARNING_RATE
 

  # Dropout probability
  keep_prob     = tf.placeholder(tf.float32)
  is_training    = tf.placeholder(tf.bool)


  # initialize parameters randomly
  X      = tf.placeholder(tf.float32, shape=[None, 720,1280,3])
  Y_     = tf.placeholder(tf.float32, shape=[None,K])
  Y_GRID = tf.placeholder(tf.float32, shape=[None,G])
  Y_BBOX = tf.placeholder(tf.float32, shape=[None,P])


  W1  = tf.get_variable("W1", shape=[12,20,3,NUM_FILTER_1], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))

  W2  = tf.get_variable("W2", shape=[3,3,NUM_FILTER_1,NUM_FILTER_2], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W3  = tf.get_variable("W3", shape=[3,3,NUM_FILTER_2,NUM_FILTER_3], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W4  = tf.get_variable("W4", shape=[3,3,NUM_FILTER_3,NUM_FILTER_4], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W5  = tf.get_variable("W5", shape=[3,3,NUM_FILTER_4,NUM_FILTER_5], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W6  = tf.get_variable("W6", shape=[3,3,NUM_FILTER_5,NUM_FILTER_6], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))

  W9  = tf.get_variable("W9", shape=[23*27*NUM_FILTER_6,NUM_NEURON_1], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W10 = tf.get_variable("W10", shape=[NUM_NEURON_1,NUM_NEURON_2], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))


  b1  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_1], dtype=tf.float32), trainable=True, name='b1')
  b2  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_2], dtype=tf.float32), trainable=True, name='b2')
  b3  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_3], dtype=tf.float32), trainable=True, name='b3')
  b4  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_4], dtype=tf.float32), trainable=True, name='b4')
  b5  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_5], dtype=tf.float32), trainable=True, name='b5')
  b6  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_6], dtype=tf.float32), trainable=True, name='b6')

  b9  = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_1], dtype=tf.float32), trainable=True, name='b9')
  b10 = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_2], dtype=tf.float32), trainable=True, name='b10')



  W_class = tf.get_variable("W_class", shape=[NUM_NEURON_2,K], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  b_class = tf.Variable(tf.constant(0.1, shape=[K], dtype=tf.float32), trainable=True, name='b_class')


  W_bbox = tf.get_variable("W_bbox", shape=[NUM_NEURON_2,P], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  b_bbox = tf.Variable(tf.constant(0.1, shape=[P], dtype=tf.float32), trainable=True, name='b_bbox')

  #W_grid = tf.get_variable("W_grid", shape=[NUM_NEURON_2,G], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W_grid = tf.get_variable("W_grid", shape=[23*27*NUM_FILTER_6,G], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  b_grid = tf.Variable(tf.constant(0.1, shape=[G], dtype=tf.float32), trainable=True, name='b_grid')






  #===== architecture =====#
  #conv1 = tf.nn.relu(tf.nn.conv2d(X,     W1, strides=[1,4,6,1], padding='VALID')+b1)
  #conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1,1,1,1], padding='SAME')+b2)
  #pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  conv1 = tf.nn.relu(tf.nn.conv2d(X,     W1, strides=[1,4,6,1], padding='SAME')+b1)
  conv2 = tf.nn.conv2d(conv1, W2, strides=[1,1,1,1], padding='SAME')
  norm1 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=is_training, renorm=True))
  pool1 = tf.nn.max_pool(norm1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
 

  #conv3 = tf.nn.relu(tf.nn.conv2d(pool1, W3, strides=[1,1,1,1], padding='SAME')+b3)
  #conv4 = tf.nn.relu(tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='SAME')+b4)
  #pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  conv3 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(pool1, W3, strides=[1,1,1,1], padding='SAME'), training=is_training, renorm=True))
  conv4 = tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='SAME')
  norm2 = tf.nn.relu(tf.layers.batch_normalization(conv4, training=is_training, renorm=True))
  pool2 = tf.nn.max_pool(norm2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
 
  #conv5 = tf.nn.relu(tf.nn.conv2d(pool2, W5, strides=[1,1,1,1], padding='SAME')+b5)
  #conv6 = tf.nn.relu(tf.nn.conv2d(conv5, W6, strides=[1,1,1,1], padding='SAME')+b6)
  #pool3 = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  conv5 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(pool2, W5, strides=[1,1,1,1], padding='SAME'), training=is_training, renorm=True))
  conv6 = tf.nn.conv2d(conv5, W6, strides=[1,1,1,1], padding='SAME')
  norm3 = tf.nn.relu(tf.layers.batch_normalization(conv6, training=is_training, renorm=True))
  pool3 = tf.nn.max_pool(norm3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



  print "conv1: ", conv1.get_shape()
  print "conv2: ", conv2.get_shape()
  print "pool1: ", pool1.get_shape()

  print "conv3: ", conv3.get_shape()
  print "conv4: ", conv4.get_shape()
  print "pool2: ", pool2.get_shape()

  print "conv5: ", conv5.get_shape()
  print "conv6: ", conv6.get_shape()
  print "pool3: ", pool3.get_shape()

  YY = tf.reshape(pool3, shape=[-1,23*27*NUM_FILTER_6])
  
  fc1 = tf.nn.relu(tf.matmul(YY,W9)+b9)
  #fc1_drop = tf.nn.dropout(fc1, keep_prob)
  
  fc2 = tf.nn.relu(tf.matmul(fc1,W10)+b10)
  #fc2_norm = tf.nn.relu(tf.layers.batch_normalization(fc2, training=is_training, renorm=True))
  #fc2_drop = tf.nn.dropout(fc2, keep_prob)
  
  Y_class = tf.matmul(fc2,W_class)+b_class
  Y_bbox  = tf.matmul(fc2,W_bbox)+b_bbox
  Y_grid  = tf.matmul(YY,W_grid)+b_grid


  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    mse_loss_bbox = tf.losses.mean_squared_error(labels=Y_BBOX, predictions=Y_bbox)
    mse_loss_grid = tf.losses.mean_squared_error(labels=Y_GRID, predictions=Y_grid)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y_class))
    reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
   

     
    total_loss = 1e-3*mse_loss_bbox + 1e-3*mse_loss_grid + 1e-3*cross_entropy + reg_loss
    #total_loss = 1e-3*mse_loss_bbox + 1e-4*mse_loss_grid + 1e-3*cross_entropy + reg_loss
    #if(np.argmax(Y_) == 0):
    #  total_loss = 1e-3*cross_entropy + reg_loss
    #else:
    #  total_loss = 1e-4*mse_loss + 1e-3*cross_entropy + reg_loss


    train_step = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9).minimize(total_loss)

    #clipping gradient to avoid gradient exploding
    #optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9)
    #gvs = optimizer.compute_gradients(total_loss)
    #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    #train_step = optimizer.apply_gradients(capped_gvs)


  correct_prediction = tf.equal(tf.argmax(Y_class, 1), tf.argmax(Y_, 1))
  correct_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Add ops to save and restore all the variables.
  var_list = tf.trainable_variables()
  g_list = tf.global_variables()
  bn_moving_mean = [g for g in g_list if 'moving_mean' in g.name]
  bn_moving_var  = [g for g in g_list if 'moving_variance' in g.name]
  var_list += bn_moving_mean
  var_list += bn_moving_var
  saver = tf.train.Saver(var_list=tf.global_variables())


  train_data_path = []
  valid_data_path = []
  mean_img = np.zeros((720,1280,3))
  for i in range(4):
    print "/home/hhwu/Reno/detection/train_%s.tfrecords" % i
    train_data_path.append("/home/hhwu/Reno/detection/train_%s.tfrecords" % i)

    print "/home/hhwu/Reno/detection/valid_%s.tfrecords" % i
    valid_data_path.append("/home/hhwu/Reno/detection/valid_%s.tfrecords" % i)

    print "/home/hhwu/Reno/detection/mean_%s.txt" % i
    m_img = np.load("/home/hhwu/Reno/detection/mean_%s.npy" % i)
    mean_img = mean_img + m_img

  mean_img = mean_img/4
  #channel_mean = np.mean(mean_img, axis=(0,1))
  #print channel_mean

  #print mean_img
  #train_data_path = "/home/hhwu/tracking/data_training/train_all.tfrecords"
  #valid_data_path = "/home/hhwu/tracking/data_training/valid_all.tfrecords"

  with tf.Session() as sess:
    ################################
    #        Training Data         #
    ################################
    train_feature = {'train/image': tf.FixedLenFeature([], tf.string),
                     'train/xmin' : tf.FixedLenFeature([], tf.int64),
                     'train/xmax' : tf.FixedLenFeature([], tf.int64),
                     'train/ymin' : tf.FixedLenFeature([], tf.int64),
                     'train/ymax' : tf.FixedLenFeature([], tf.int64),
                     'train/grid' : tf.FixedLenFeature([G], tf.float32),
                     #'train/grid' : tf.VarLenFeature(tf.float32),
                     'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    train_filename_queue = tf.train.string_input_producer(train_data_path, shuffle=True)
    #train_filename_queue = tf.train.string_input_producer([train_data_path])
    #filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    train_reader = tf.TFRecordReader()
    _, train_serialized_example = train_reader.read(train_filename_queue)

        # Decode the record read by the reader
    train_features = tf.parse_single_example(train_serialized_example, features=train_feature)
    # Convert the image data from string back to the numbers
    #train_image = tf.decode_raw(train_features['train/image'], tf.uint8)
    train_image = tf.cast(tf.decode_raw(train_features['train/image'], tf.uint8), tf.float32)
    
    # Cast label data into int32
    train_label_idx = tf.cast(train_features['train/label'], tf.int32)
    train_label = tf.one_hot(train_label_idx, K)

    train_label_xmin = tf.cast(train_features['train/xmin'], tf.float32)
    train_label_xmax = tf.cast(train_features['train/xmax'], tf.float32)
    train_label_ymin = tf.cast(train_features['train/ymin'], tf.float32)
    train_label_ymax = tf.cast(train_features['train/ymax'], tf.float32)
    train_label_grid = tf.cast(train_features['train/grid'], tf.float32)
    #train_label_grid = tf.sparse_tensor_to_dense(train_features['train/grid'], default_value=0)
    #train_label_grid.set_shape([1,G])


    # Reshape image data into the original shape
    train_image = tf.reshape(train_image, [720, 1280, 3])
    train_image = tf.subtract(train_image,mean_img)
    # train_image = tf.image.per_image_standardization(train_image)

    #train_image = tf.image.resize_images(train_image, [640, 640])
    #train_image = tf.image.random_flip_left_right(train_image)


    sel = np.random.uniform(0,1)
    if(sel <= 0.5):
      train_image = tf.image.flip_left_right(train_image)
      train_label_box_coor = tf.stack([1279-train_label_xmax, 1279-train_label_xmin, train_label_ymin, train_label_ymax])
    #elif(sel <= 0.66):
    #  train_image = tf.image.flip_up_down(train_image)
    #  train_label_box_coor = tf.stack([train_label_xmin, train_label_xmax, 719-train_label_ymax, 719-train_label_ymin])
    else:
      train_label_box_coor = tf.stack([train_label_xmin, train_label_xmax, train_label_ymin, train_label_ymax])

    #print "TFRecord: hhwu !"
    train_images, train_labels, tr_box_coors, train_grids = tf.train.batch([train_image, train_label, train_label_box_coor, train_label_grid], 
                                                 batch_size=mini_batch, capacity=20*mini_batch, num_threads=16)
    #train_images, train_labels = tf.train.batch([train_image, train_label], 
    #                                             batch_size=mini_batch, capacity=20*mini_batch, num_threads=16)


    ################################
    #       Validation Data        #
    ################################
    valid_feature = {'valid/image': tf.FixedLenFeature([], tf.string),
                     'valid/xmin' : tf.FixedLenFeature([], tf.int64),
                     'valid/xmax' : tf.FixedLenFeature([], tf.int64),
                     'valid/ymin' : tf.FixedLenFeature([], tf.int64),
                     'valid/ymax' : tf.FixedLenFeature([], tf.int64),
                     'valid/grid' : tf.FixedLenFeature([G], tf.float32),
                     'valid/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    #valid_filename_queue = tf.train.string_input_producer([valid_data_path])
    valid_filename_queue = tf.train.string_input_producer(valid_data_path)
    #filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    valid_reader = tf.TFRecordReader()
    _, valid_serialized_example = valid_reader.read(valid_filename_queue)

        # Decode the record read by the reader
    valid_features = tf.parse_single_example(valid_serialized_example, features=valid_feature)
    # Convert the image data from string back to the numbers
    valid_image = tf.cast(tf.decode_raw(valid_features['valid/image'], tf.uint8), tf.float32)
    
    # Cast label data into int32
    valid_label_idx = tf.cast(valid_features['valid/label'], tf.int32)
    valid_label = tf.one_hot(valid_label_idx, K)
    valid_label_xmin = tf.cast(valid_features['valid/xmin'], tf.float32)
    valid_label_xmax = tf.cast(valid_features['valid/xmax'], tf.float32)
    valid_label_ymin = tf.cast(valid_features['valid/ymin'], tf.float32)
    valid_label_ymax = tf.cast(valid_features['valid/ymax'], tf.float32)
    valid_label_grid = tf.cast(valid_features['valid/grid'], tf.float32)
    #valid_label_grid = tf.sparse_tensor_to_dense(valid_features['valid/grid'], default_value=0)
    #valid_label_grid.set_shape([1,G])


    # Reshape image data into the original shape
    valid_image = tf.reshape(valid_image, [720, 1280, 3])
    valid_image = tf.subtract(valid_image,mean_img)
    # valid_image = tf.image.per_image_standardization(valid_image)

    #valid_image = tf.image.resize_images(valid_image, [640, 640])

    valid_label_box_coor = tf.stack([valid_label_xmin, valid_label_xmax, valid_label_ymin, valid_label_ymax])
    

    valid_images, valid_labels, vl_box_coors, valid_grids = tf.train.batch([valid_image, valid_label, valid_label_box_coor, valid_label_grid], 
                                                 batch_size=100, capacity=1000, num_threads=16)



    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


    ###############################
    # Restore variables from disk #
    ###############################
    #model_name = "model_small_0.31_97000"
    #saver.restore(sess, "./checkpoint/%s.ckpt" % model_name)
    #print "Model %s restored." % model_name


    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #x, y = batchRead(class_name, mean_img, pool)


    highest_IOU = 0
    for itr in xrange(100000):
      #print "bn_moving_vars: ", bn_moving_vars
      x, y, box_coord, grid = sess.run([train_images, train_labels, tr_box_coors, train_grids])
      #print grid
      #print grid.shape
      #train_step.run(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, keep_prob: DROPOUT_PROB, is_training: True})
      sess.run([train_step, update_ops], feed_dict={X: x, Y_: y, Y_BBOX: box_coord, Y_GRID: grid, keep_prob: DROPOUT_PROB, is_training: True})

      #elapsed_time = time.time() - start_time
      #print "Time for training: %f" % elapsed_time
      if itr % 20 == 0:
        #for g in bn_moving_mean:
        #  print "bn_moving_mean: ", g.eval()

        #for g in bn_moving_var:
        #  print "bn_moving_var: ", g.eval()
       

        pred_bbox = Y_bbox.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, Y_GRID: grid, keep_prob: 1.0, is_training: False})
        #print "pred_bbox: ", pred_bbox
        #tt = np.mean(checkIOU(box_coord, pred_bbox))
        #print tt
        #print tt.shape
        print "Iter %d:  learning rate: %f  cross entropy: %.3f  mse_bbox: %.3f  mse_grid: %.3f reg: %.3f  accuracy: %.3f  mean IOU: %.3f" % (itr,
                                              LEARNING_RATE,
                                              #lr.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord}),
                                              cross_entropy.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, Y_GRID: grid, keep_prob: 1.0, is_training: False}),
                                              mse_loss_bbox.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, Y_GRID: grid, keep_prob: 1.0, is_training: False}),
                                              mse_loss_grid.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, Y_GRID: grid, keep_prob: 1.0, is_training: False}),
                                              reg_loss.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, Y_GRID: grid, keep_prob: 1.0, is_training: False}),
                                              accuracy.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, Y_GRID: grid, keep_prob: 1.0, is_training: False}),
                                              np.mean(checkIOU(box_coord, pred_bbox)))
        #for j in range(0, mini_batch):
        #  bbox_image = drawBBox(x[j],pred_bbox[j], box_coord[j])
        #  
        #  if np.argmax(y[j]) == 0:
        #      print "background"
        #      io.imsave("%s_%05d_%s.%s" % ("./train_images/img", 100*i+j, 'bg', 'jpg'), bbox_image/256.0)
        #  else:
        #      print "car"
        #      io.imsave("%s_%05d_%s.%s" % ("./train_images/img", 100*i+j, 'car', 'jpg'), bbox_image/256.0)




 
  
      if itr % 1000 == 0 and itr != 0:
      #if itr % 50 == 0:
        valid_accuracy = 0.0
        valid_IOU = 0.0
        for i in range(0,20):
          test_x, test_y, box_coord, grid = sess.run([valid_images, valid_labels, vl_box_coors, valid_grids])
          #Y_labels_with_grid = expandLabel(test_y, box_coord, 100)
          #box_coord[:,0] = box_coord[:,0]
          #box_coord[:,1] = box_coord[:,1]
          #box_coord[:,2] = box_coord[:,2]
          #box_coord[:,3] = box_coord[:,3]

          pred_bbox = Y_bbox.eval(feed_dict={X: test_x, Y_: test_y, Y_BBOX: box_coord, Y_GRID: grid, keep_prob: 1.0, is_training: False})

   
          valid_accuracy += correct_sum.eval(feed_dict={X: test_x, Y_: test_y, Y_BBOX: box_coord, Y_GRID: grid, keep_prob: 1.0, is_training: False})
          valid_IOU += np.mean(checkIOU(box_coord, pred_bbox))

        valid_accuracy = valid_accuracy/2000
        valid_IOU = valid_IOU/20

        print "Validation Accuracy: %f " %  valid_accuracy
        print "Validation Mean IOU: %f " %  valid_IOU
        #valid_result.write("Validation Accuracy: %f" % (valid_accuracy/20000))
        #valid_result.write("\n")
        if valid_IOU > highest_IOU:
          highest_IOU = valid_IOU
          model_name = "./checkpoint/model_small_%.2f_%d.ckpt" % (valid_IOU, itr)
          save_path = saver.save(sess, model_name)
          print("Model saved in file: %s" % save_path)

       

      #if itr % 10000 == 0 and itr != 0:
      #  LEARNING_RATE = 0.1*LEARNING_RATE
      #  model_name = "./checkpoint/model_small_%d.ckpt" % itr
      #  save_path = saver.save(sess, model_name)
      #  #save_path = saver.save(sess, "./checkpoint/model.ckpt")
      #  print("Model saved in file: %s" % save_path)




    #model_name = "./checkpoint/model_small_trained.ckpt"
    #save_path = saver.save(sess, model_name)
    ##save_path = saver.save(sess, "./checkpoint/model.ckpt")
    #print("Model saved in file: %s" % save_path)


    coord.request_stop()
    coord.join(threads)
    sess.close()



