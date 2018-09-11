#!/usr/bin/python

import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np
import sys
import csv
import os
from skimage import io
from multiprocessing.pool import ThreadPool


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
      value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def xmlParser(f_path):
  tree = ET.parse(f_path)
  xmin = -1
  xmax = -1
  ymin = -1
  ymax = -1
  name = 0


  infile = open("label.txt", "r")
  lines = infile.readlines()

  lines = map(lambda s: s.strip(), lines)
  label_dict = {}
  for l in lines:
    elements = l.split()
    label_dict[elements[0]] = elements[1]


  for elem in tree.iter():
    tag = elem.tag
    if tag == 'xmin':
      xmin = int(elem.text)
    elif tag == 'xmax':
      xmax = int(elem.text)
    elif tag == 'ymin':
      ymin = int(elem.text)
    elif tag == 'ymax':
      ymax = int(elem.text)
    elif tag == 'name':
      name = label_dict[elem.text]

  if name == '0':
    #print "bg"
    return name, -1, -1, -1, -1;
  else:
    #print "car"
    return name, xmin, xmax, ymin, ymax;


def checkIOU(label_BBox, pred_BBox):
  #print "label_BBox shape: ", label_BBox.shape
  #print "pred_BBox shape: ", pred_BBox.shape

  IOU = np.zeros(label_BBox.shape[0])

  ###############################
  #  check validity of pred box #
  #   (xmin, xmax, ymin, ymax)  #
  ###############################
  if checkIntersection(label_BBox, pred_BBox) == 1:

    xmin_A = pred_BBox[0]
    xmax_A = pred_BBox[1]
    ymin_A = pred_BBox[2]
    ymax_A = pred_BBox[3]


    xmin_B = label_BBox[0]
    xmax_B = label_BBox[1]
    ymin_B = label_BBox[2]
    ymax_B = label_BBox[3]

    xmin_intersection = np.maximum(xmin_A, xmin_B)
    xmax_intersection = np.minimum(xmax_A, xmax_B)
    ymin_intersection = np.maximum(ymin_A, ymin_B)
    ymax_intersection = np.minimum(ymax_A, ymax_B)

    intersection_width  = xmax_intersection-xmin_intersection
    intersection_height = ymax_intersection-ymin_intersection

    if intersection_width < 0 or intersection_height < 0:
      IOU = 0
    else:
      intersection_area = intersection_width*intersection_height
      area_two_boxes = (xmax_A-xmin_A)*(ymax_A-ymin_A) + (xmax_B-xmin_B)*(ymax_B-ymin_B)
      IOU = intersection_area/(area_two_boxes-intersection_area) 

  else:
    IOU = 0
      

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



def IoU_grid(label_BBox, grid_y, grid_x):
  grid_BBox = np.zeros(4)
  iou       = np.zeros(grid_y*grid_x)
  x_unit = 1280/grid_x
  y_unit = 720/grid_y

  for i in range(grid_x):
    for j in range(grid_y):
      grid_BBox[0] = i*x_unit
      grid_BBox[1] = (i+1)*x_unit - 1
      grid_BBox[2] = j*y_unit - 1
      grid_BBox[3] = (j+1)*y_unit - 1
      iou[i*grid_y+j] = checkIOU(label_BBox, grid_BBox)

  return iou
  

if __name__ == '__main__':
  #datapath = '/home/hhwu/tensorflow_work/TX2_tracking/data_training/bird1'
  datapath = sys.argv[1]
  print "Path: ", datapath

  grid_x = 16
  grid_y = 16


  file_list = []
  for dirpath, dirnames, filenames in os.walk(datapath):
    #print "dirpath: ", dirpath
    #print "dirnames: ", dirnames
    #print "The number of files: %d" % len(filenames)
    
    file_list = filenames

  image_list = []
  xml_list = []

  for f in file_list:
    if f.endswith(".xml"):
      xml_list.append(f)
    elif f.endswith(".jpg"):
      image_list.append(f)

  image_list = sorted(image_list)
  xml_list = sorted(xml_list)
  assert len(image_list) == len(xml_list)




  #for xml_elem in xml_list:

  order_idx = np.random.randint(0,len(xml_list),len(xml_list))
  train_idx = order_idx[1120:]
  valid_idx = order_idx[:1120]


  output_name = "train_%s.tfrecords" % sys.argv[2]
  mean_name = "mean_%s" % sys.argv[2]
  writer = tf.python_io.TFRecordWriter(output_name)
  mean_img = np.zeros((720, 1280, 3))
  label_BBox = np.zeros(4)

  for i in train_idx:
    xml_f_path = "%s/%s" % (datapath, xml_list[i])
    jpg_f_path = "%s/%s" % (datapath, image_list[i])
    #print "Path: ", f_path
 
    label, xmin, xmax, ymin, ymax = xmlParser(xml_f_path)
    target_img = io.imread(jpg_f_path)

    label_BBox[0] = xmin
    label_BBox[1] = xmax
    label_BBox[2] = ymin
    label_BBox[3] = ymax
    grid = IoU_grid(label_BBox, grid_y, grid_x)

    mean_img = mean_img+target_img

    #print grid
    #print "label: ", label
    #print "xmin: ", xmin
    #print "xmax: ", xmax
    #print "ymin: ", ymin
    #print "ymax: ", ymax

    feature = {'train/label': _int64_feature(int(label)),
               'train/xmin' : _int64_feature(int(xmin)),
               'train/xmax' : _int64_feature(int(xmax)),
               'train/ymin' : _int64_feature(int(ymin)),
               'train/ymax' : _int64_feature(int(ymax)),
               'train/grid' : _float_feature(grid.tolist()),
               'train/image': _bytes_feature(tf.compat.as_bytes(target_img.tostring()))}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

  print "File %s is written." % output_name
  np.save(mean_name, mean_img/len(train_idx))


  output_name = "valid_%s.tfrecords" % sys.argv[2]
  writer = tf.python_io.TFRecordWriter(output_name)
  for i in valid_idx:
    xml_f_path = "%s/%s" % (datapath, xml_list[i])
    jpg_f_path = "%s/%s" % (datapath, image_list[i])
    #print "Path: ", f_path
 
    label, xmin, xmax, ymin, ymax = xmlParser(xml_f_path)

    label_BBox[0] = xmin
    label_BBox[1] = xmax
    label_BBox[2] = ymin
    label_BBox[3] = ymax
    grid = IoU_grid(label_BBox, grid_y, grid_x)


    target_img = io.imread(jpg_f_path)

    #print "label: ", label
    #print "xmin: ", xmin
    #print "xmax: ", xmax
    #print "ymin: ", ymin
    #print "ymax: ", ymax

    feature = {'valid/label': _int64_feature(int(label)),
               'valid/xmin' : _int64_feature(int(xmin)),
               'valid/xmax' : _int64_feature(int(xmax)),
               'valid/ymin' : _int64_feature(int(ymin)),
               'valid/ymax' : _int64_feature(int(ymax)),
               'valid/grid' : _float_feature(grid.tolist()),
               'valid/image': _bytes_feature(tf.compat.as_bytes(target_img.tostring()))}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

  print "File %s is written." % output_name



