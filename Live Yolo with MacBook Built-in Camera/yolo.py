import numpy as np
from absl import flags
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import sys
from PIL import Image


ROOT = "./"
filename_darknet_weights=ROOT+'yolov3.weights'
filename_classes=ROOT+'coco.names'
filename_converted_weights = ROOT+'yolov3.tf'

# Flags are used to define several options for YOLO.
flags.DEFINE_string('classes', filename_classes, 'path to classes file')
flags.DEFINE_string('weights', filename_converted_weights, 'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
FLAGS([sys.argv[0]])

yolo = YoloV3(classes=FLAGS.num_classes)

# Load weights and classes
yolo.load_weights(FLAGS.weights).expect_partial()
print('weights loaded')
class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
print('classes loaded')


def predict(img):
    arr = tf.expand_dims(img, 0)
    arr = transform_images(arr, FLAGS.size)
    FLAGS.yolo_score_threshold = 0.5
    boxes, scores, classes, nums = yolo(arr)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    return img
