from __future__ import print_function
from datetime import datetime
import time
import math
import tensorflow as tf
slim = tf.contrib.slim


# # truncated normal distribution
# 

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


# # inception network default arguments
# 

def inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1, 
                                   batch_norm_var_collection='moving_vars'):
    
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }
    
    # slim.arg_scope automatically assigns default values to parameters
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                       weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d],
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           activation_fn=tf.nn.relu,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=batch_norm_params) as sc:
            return sc


# # inception-v3 function
# 

def inception_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 reuse=None,
                 scope='inceptionV3'):
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.variable_scope(scope, 'inceptionV3', [inputs, num_classes], reuse=reuse) as scope:
    with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.dropout]):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
        # 299 x 299 x 3
        end_points['conv0'] = slim.conv2d(inputs, 32, [3, 3], stride=2,
                                         scope='conv0')
        # 149 x 149 x 32
        end_points['conv1'] = slim.conv2d(end_points['conv0'], 32, [3, 3],
                                         scope='conv1')
        # 147 x 147 x 32
        end_points['conv2'] = slim.conv2d(end_points['conv1'], 64, [3, 3],
                                         padding='SAME', scope='conv2')
        # 147 x 147 x 64
        end_points['pool1'] = slim.max_pool2d(end_points['conv2'], [3, 3],
                                           stride=2, scope='pool1')
        # 73 x 73 x 64
        end_points['conv3'] = slim.conv2d(end_points['pool1'], 80, [1, 1],
                                         scope='conv3')
        # 73 x 73 x 80.
        end_points['conv4'] = slim.conv2d(end_points['conv3'], 192, [3, 3],
                                         scope='conv4')
        # 71 x 71 x 192.
        end_points['pool2'] = slim.max_pool2d(end_points['conv4'], [3, 3],
                                           stride=2, scope='pool2')
        # 35 x 35 x 192.
        net = end_points['pool2']
      
     # Inception blocks
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
        
        # mixed: 35 x 35 x 256.
        with tf.variable_scope('mixed_35x35x256a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = slim.conv2d(net, 48, [1, 1])
            branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 32, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x256a'] = net
        
        # mixed_1: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = slim.conv2d(net, 48, [1, 1])
            branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x288a'] = net
        
        # mixed_2: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = slim.conv2d(net, 48, [1, 1])
            branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x288b'] = net
        
        # mixed_3: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3],
                                      stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_17x17x768a'] = net
        
        # mixed4: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 128, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 128, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 128, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768b'] = net
        
        # mixed_5: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768c'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 160, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 160, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768c'] = net
        
        # mixed_6: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768d'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 160, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 160, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768d'] = net
        
        # mixed_7: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768e'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 192, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 192, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 192, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768e'] = net
        
        # Auxiliary Head logits
        aux_logits = tf.identity(end_points['mixed_17x17x768e'])
        with tf.variable_scope('aux_logits'):
          aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                    padding='VALID')
          aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='proj')
          # Shape of feature map before the final layer.
          shape = aux_logits.get_shape()
          aux_logits = slim.conv2d(aux_logits, 768, shape[1:3], 
                               weights_initializer=trunc_normal(0.01), padding='VALID')
          aux_logits = slim.flatten(aux_logits)
          aux_logits = slim.fully_connected(aux_logits, num_classes, activation_fn=None,
                               weights_initializer=trunc_normal(0.01))
          end_points['aux_logits'] = aux_logits
        
        # mixed_8: 8 x 8 x 1280.
        # Note that the scope below is not changed to not void previous
        # checkpoints.
        # (TODO) Fix the scope when appropriate.
        with tf.variable_scope('mixed_17x17x1280a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 192, [1, 1])
            branch3x3 = slim.conv2d(branch3x3, 320, [3, 3], stride=2,
                                   padding='VALID')
          with tf.variable_scope('branch7x7x3'):
            branch7x7x3 = slim.conv2d(net, 192, [1, 1])
            branch7x7x3 = slim.conv2d(branch7x7x3, 192, [1, 7])
            branch7x7x3 = slim.conv2d(branch7x7x3, 192, [7, 1])
            branch7x7x3 = slim.conv2d(branch7x7x3, 192, [3, 3],
                                     stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])
          end_points['mixed_17x17x1280a'] = net
        
        # mixed_9: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat(axis=3, values=[slim.conv2d(branch3x3, 384, [1, 3]),
                                                  slim.conv2d(branch3x3, 384, [3, 1])])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 448, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat(axis=3, values=[slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     slim.conv2d(branch3x3dbl, 384, [3, 1])])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_8x8x2048a'] = net
        
        # mixed_10: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat(axis=3, values=[slim.conv2d(branch3x3, 384, [1, 3]),
                                                  slim.conv2d(branch3x3, 384, [3, 1])])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 448, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat(axis=3, values=[slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                                     slim.conv2d(branch3x3dbl, 384, [3, 1])])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_8x8x2048b'] = net
        
        # Final pooling and prediction
        with tf.variable_scope('logits'):
          shape = net.get_shape()
          net = slim.avg_pool2d(net, shape[1:3], padding='VALID', scope='pool')
          # 1 x 1 x 2048
          net = slim.dropout(net, dropout_keep_prob, scope='dropout')
          net = slim.flatten(net, scope='flatten')
          # 2048
          logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='logits')
          # 1000
          end_points['logits'] = logits
          end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      return logits, end_points


# # time run function
# 

def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                     (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' %
         (datetime.now(), info_string, num_batches, mn, sd))


# # train
# 

batch_size = 32
height, width = 299, 299
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(inception_v3_arg_scope()):
    logits, end_points =  inception_v3(inputs, is_training=False)
    
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches=100
time_tensorflow_run(sess, logits, 'Forward')


import os 
import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt 


# # convert images to bytes
# 

cwd=os.getcwd()
classes={'car','horse'} 
writer= tf.python_io.TFRecordWriter("car_horse.tfrecords") 

for index,name in enumerate(classes):
    class_path=cwd+'/'+name+'/'
    for img_name in os.listdir(class_path): 
        img_path=class_path+img_name 
        img=Image.open(img_path)
        img= img.resize((128,128))
        img_raw=img.tobytes()
        #plt.imshow(img) # if you want to check you image,please delete '#'
        #plt.show()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) 
        writer.write(example.SerializeToString()) 

writer.close()


# # read tfrecords func
# 

def read_and_decode(filename): # read iris_contact.tfrecords
    filename_queue = tf.train.string_input_producer([filename])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
               features={'label': tf.FixedLenFeature([], tf.int64),
               'img_raw' : tf.FixedLenFeature([], tf.string),})#return image and label

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])  #reshape image to 512*80*3
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #throw img tensor
    label = tf.cast(features['label'], tf.int32) #throw label tensor
    return img, label


# # convert tfrecords to image
# 

filename_queue = tf.train.string_input_producer(["car_horse.tfrecords"]) 
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #return file and file_name
features = tf.parse_single_example(serialized_example,
           features={'label': tf.FixedLenFeature([], tf.int64),
                     'img_raw' : tf.FixedLenFeature([], tf.string),})  
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [128, 128, 3])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess: 
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(5):
        example, l = sess.run([image,label])#take out image and label
        img=Image.fromarray(example, 'RGB')
        img.save(cwd+str(i)+'_Label_'+str(l)+'.jpg')#save image
        print(example, l, example.shape)
    coord.request_stop()
    coord.join(threads)


import tensorflow as tf
import glob
from itertools import groupby
from collections import defaultdict


sess = tf.InteractiveSession()
image_filenames = glob.glob("./dataset/StanfordDogs/n02*/*.jpg")
image_filenames[0:2]
training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)
image_filename_with_breed = map(lambda filename: (filename.split("/")[2], 
                                                 filename), image_filenames)
for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    for i, breed_image in enumerate(breed_images):
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])
    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])
    breed_training_count_float = float(breed_training_count)
    breed_testing_count_float = float(breed_testing_count)
    assert round(breed_testing_count_float / (breed_training_count_float +         breed_testing_count_float), 2) > 0.18, "Not enough testing images."
print("------------training_dataset testing_dataset END --------------------")
print(len(testing_dataset))
print(len(training_dataset))


# # convert images to bytes func
# 

def write_records_file(dataset, record_location):
    writer = None
    current_index = 0
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)
                print ("----------------------"+record_filename + "---------------------------") 
            current_index += 1
            image_file = tf.read_file(image_filename)
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, [250, 151])
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            image_label = breed.encode("utf-8")
            example = tf.train.Example(features=tf.train.Features(feature={
              'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
              'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))
            writer.write(example.SerializeToString())
    #writer.close()


# # write tfrecords
# 

#write_records_file(testing_dataset, "./result/test/testing-image")
write_records_file(training_dataset, "./result/train/training-image")
print("------------------write_records_file testing_dataset training_dataset END-------------------")
filename_queue = tf.train.string_input_producer(
tf.train.match_filenames_once("./result/test/*.tfrecords"))


# # load images from tfrecords
# 

reader = tf.TFRecordReader()
_, serialized = reader.read(filename_queue)
features = tf.parse_single_example(
serialized,
    features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    })
record_image = tf.decode_raw(features['image'], tf.uint8)
image = tf.reshape(record_image, [250, 151, 1])
label = tf.cast(features['label'], tf.string)
min_after_dequeue = 10
batch_size = 3
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
print("---------------------load image from TFRecord END----------------------")


# # conv and pool
# 

float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)
conv2d_layer_one = tf.contrib.layers.convolution2d(
    float_image_batch,
    num_outputs=32,
    kernel_size=(5,5),
    activation_fn=tf.nn.relu,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    stride=(2, 2),
    trainable=True)
pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
conv2d_layer_one.get_shape(), pool_layer_one.get_shape()
print("--------------------------------conv2d_layer_one pool_layer_one END--------------------------------")
conv2d_layer_two = tf.contrib.layers.convolution2d(
    pool_layer_one,
    num_outputs=64,
    kernel_size=(5,5),
    activation_fn=tf.nn.relu,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    stride=(1, 1),
    trainable=True)
pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
conv2d_layer_two.get_shape(), pool_layer_two.get_shape()
print("-----------------------------conv2d_layer_two pool_layer_two END---------------------------------")


# # flattend and fully connected layer
# 

flattened_layer_two = tf.reshape(pool_layer_two, [batch_size, -1])
flattened_layer_two.get_shape()
print("----------------------------------------flattened_layer_two END-----------------------------------------")
hidden_layer_three = tf.contrib.layers.fully_connected(
    flattened_layer_two, 512,
    weights_initializer=lambda i, dtype, partition_info=None: tf.truncated_normal([38912, 512], stddev=0.1),
    activation_fn=tf.nn.relu)
hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)
final_fully_connected = tf.contrib.layers.fully_connected(
    hidden_layer_three,
    120,
    weights_initializer=lambda i, dtype, partition_info=None: tf.truncated_normal([512, 120], stddev=0.1))
print("-----------------------final_fully_connected END--------------------------------------")


# # train and predict
# 

labels = list(map(lambda c: c.split("/")[-1], glob.glob("./dataset/StanfordDogs/*")))
train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l))[0,0:1][0], label_batch, dtype=tf.int64)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_fully_connected, labels=train_labels))
batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, batch * 3, 120, 0.95, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
train_prediction = tf.nn.softmax(final_fully_connected)
print(train_prediction)
print("--------------------------------train_prediction END---------------------------------------")
filename_queue.close(cancel_pending_enqueues=True)
print("-------------------------------END---------------------------")





import tensorflow as tf


# # features function
# 

def input_fn():
    return {'example_id': tf.constant(['1', '2', '3']),
                'feature1': tf.constant([[0.0], [1.0], [3.0]]),
                'feature2': tf.constant([[0.0], [-1.2], [1.0]]),}, tf.constant([[1], [0], [1]])

feature1 = tf.contrib.layers.real_valued_column('feature1')
feature2 = tf.contrib.layers.real_valued_column('feature2')


# # classifier and evaluate
# 

svm_classifier = tf.contrib.learn.SVM(feature_columns=[feature1, feature2],
                                     example_id_column='example_id',
                                     l1_regularization=0.0, l2_regularization=0.0)

svm_classifier.fit(input_fn=input_fn, steps=30)
metrics = svm_classifier.evaluate(input_fn=input_fn, steps=1)
loss = metrics['loss']
accuracy = metrics['accuracy']
print('loss: ', loss)
print('accuracy: ', accuracy)


import tensorflow as tf


# # features and my_metric function
# 

def _input_fn_train():
    target = tf.constant([[1], [0], [0], [0]])
    features = {'x': tf.ones(shape=[4, 1], dtype=tf.float32),}
    return features, target

def _my_metric_op(predictions, targets):
    predictions = tf.slice(predictions, [0, 1], [-1, 1])
    return tf.reduce_sum(tf.matmul(predictions, targets))


# # classifier
# 

classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=[tf.contrib.layers.real_valued_column('x')],
    hidden_units=[3, 3],
    config=tf.contrib.learn.RunConfig(tf_random_seed=1))


# # train and test
# 

classifier.fit(input_fn=_input_fn_train, steps=100)

scores = classifier.evaluate(
    input_fn=_input_fn_train,
    steps=100,
    metrics={'my_accuracy': tf.contrib.metrics.streaming_accuracy,
             ('my_precision', 'classes'): tf.contrib.metrics.streaming_precision,
             ('my_metric', 'probabilities'): _my_metric_op})

print("evaluation scores: ", scores)


