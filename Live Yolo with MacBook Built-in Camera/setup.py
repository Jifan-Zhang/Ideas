import sys
import os
import tensorflow as tf
import requests
os.system(f"{sys.executable} -m pip install git+https://github.com/zzh8829/yolov3-tf2.git@master")

ROOT = "./yolo/"

f=open(ROOT+"yolov3.weights",'wb')
f.write( requests.get('https://pjreddie.com/media/files/yolov3.weights').content)
f.close()
print('weights done')

f=open(ROOT+"convert.py",'wb')
f.write( requests.get('https://raw.githubusercontent.com/zzh8829/yolov3-tf2/master/convert.py').content)
f.close()
print('convert done')

f=open(ROOT+"coco.names",'wb')
f.write( requests.get('https://raw.githubusercontent.com/zzh8829/yolov3-tf2/master/data/coco.names').content)
f.close()
print('names done')

filename_darknet_weights = tf.keras.utils.get_file(
    os.path.join(ROOT,'yolov3.weights'),
    origin='https://pjreddie.com/media/files/yolov3.weights')
TINY = False

filename_convert_script = tf.keras.utils.get_file(
    os.path.join(os.getcwd(),'convert.py'),
    origin='https://raw.githubusercontent.com/zzh8829/yolov3-tf2/master/convert.py')

filename_classes = tf.keras.utils.get_file(
    os.path.join(ROOT,'coco.names'),
    origin='https://raw.githubusercontent.com/zzh8829/yolov3-tf2/master/data/coco.names')
filename_converted_weights = os.path.join(ROOT,'yolov3.tf')


filename_darknet_weights=ROOT+'yolov3.weights'
filename_convert_script=ROOT+'convert.py'
filename_classes=ROOT+'coco.names'
filename_converted_weights = ROOT+'yolov3.tf'

os.system(f"{sys.executable} {filename_convert_script} --weights {filename_darknet_weights} --output {filename_converted_weights}")
os.remove(filename_convert_script)
