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
  #  elif tag == 'name':
  #    name = label_dict[elem.text]

  #if name == '0':
  #  #print "bg"
  #  return name, -1, -1, -1, -1;
  #else:
  #  #print "car"
  #  return name, xmin, xmax, ymin, ymax;

  return '1', xmin, xmax, ymin, ymax;

if __name__ == '__main__':
  #datapath = '/home/hhwu/tensorflow_work/TX2_tracking/data_training/bird1'
  datapath = sys.argv[1]
  print "Path: ", datapath


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

  order_idx = range(0,len(xml_list))
  #train_idx = order_idx[600:]
  #valid_idx = order_idx[:600]


  output_name = "test_%s.tfrecords" % sys.argv[2]
  writer = tf.python_io.TFRecordWriter(output_name)
  for i in order_idx:
    xml_f_path = "%s/%s" % (datapath, xml_list[i])
    jpg_f_path = "%s/%s" % (datapath, image_list[i])
    #print "Path: ", f_path
 
    label, xmin, xmax, ymin, ymax = xmlParser(xml_f_path)
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
               'valid/image': _bytes_feature(tf.compat.as_bytes(target_img.tostring()))}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

  print "File %s is written." % output_name

