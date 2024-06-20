# <a id='sectionTop'></a>
# # Image Classification with Tensorflow/Inception
# 
# [1. Installation (Mac)](#section1)<br>
# &nbsp;&nbsp;&nbsp;A. Tensorflow<br>
# &nbsp;&nbsp;&nbsp;B. Docker Container<br>
# &nbsp;&nbsp;&nbsp;C. Inception v3<br>
# [2. Prep of images/data for tensorflow](#section2)<br>
# &nbsp;&nbsp;&nbsp;A. Flowers Data (Google "Tensorflow for Poets" Demo)<br>
# [3. Retraining the model](#section3)<br>
# [4. Validation code for classification model (label_image.py)](#section4)<br>
# 

# <a id='section1'></a><span style="float:right">[Back to Top](#sectionTop)</span>
# # 1. Installation (Mac)
# <br>Install Docker for Mac
# <br>(other OS details here: https://docs.docker.com/engine/installation/)
# <div class="alert alert-block alert-info">
# https://download.docker.com/mac/stable/Docker.dmg
# <br>Run DMG file
# <br>Drag Docker.app to Applications folder
# <br>Run Docker App
# </div>
# 
# #### Rest of installation done from command line (Mac Terminal, Assumes Python 3.6 installed)
# Install Tensorflow (https://www.tensorflow.org/install/)
# <div class="alert alert-block alert-info">
# pip3 install tensorflow
# </div>
# 
# Validate Install - Start python shell, then run 4-line code snippet, prints "Hello, TensorFlow!" if it's working
# <div class="alert alert-block alert-info">
# python
# <br>
# <br>import tensorflow as tf
# <br>hello = tf.constant('Hello, TensorFlow!')
# <br>sess = tf.Session()
# <br>print(sess.run(hello))
# </div>
# 
# Install Docker Container for Tensorflow 1.1.0 (just download/install, exit, and remove it for now)
# <div class="alert alert-block alert-info">
# docker run -it --name tensorflow gcr.io/tensorflow/tensorflow:1.1.0 bash
# <br>exit
# <br>docker rm -f tensorflow
# </div>
# 
# Copy Inception v3 architecture and pre-trained library
# <div class="alert alert-block alert-info">
# git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
# <br>cd tensorflow-for-poets-2
# </div>
# 

# <a id='section2'></a><span style="float:right">[Back to Top](#sectionTop)</span>
# # 2. Prep of images/data for tensorflow
# 
# Tensorflow appears to prefer JPGs. PNG can supposedly work (as it does in Tensorbox demo), but I haven't gotten it working for this basic classification example.
# 

# ## Flowers Data (Google "Tensorflow for Poets" Demo):
# <div class="alert alert-block alert-info">
# wget http://download.tensorflow.org/example_images/flower_photos.tgz
# <br>tar -C tf_files/ -xvzf flower_photos.tgz
# <br>ls tf_files/flower_photos
# </div>
# 
# Should list 5 directories, which have ~3600 images in total:
# <div class="alert alert-block alert-info">
# daisy/
# <br>dandelion/
# <br>roses/
# <br>sunflowers/
# <br>tulips/
# <br>LICENSE.txt
# </div>
# 
# Move a few images of each flower out of training location to use for testing later:
# <div class="alert alert-block alert-info">
# mkdir test_images
# <br>mkdir test_images/flower_photos
# <br>mv ./tf_files/flower_photos/roses/102501987_3cdb8e5394_n.jpg ./test_images/flower_photos/rose1.jpg
# <br>mv ./tf_files/flower_photos/roses/6231418894_7946a7712b_n.jpg ./test_images/flower_photos/rose2.jpg
# <br>mv ./tf_files/flower_photos/roses/21522100663_455b77a90c_n.jpg ./test_images/flower_photos/rose3.jpg
# <br>mv ./tf_files/flower_photos/daisy/5547758_eea9edfd54_n.jpg ./test_images/flower_photos/daisy1.jpg
# <br>mv ./tf_files/flower_photos/daisy/2713919471_301fcc941f.jpg ./test_images/flower_photos/daisy2.jpg
# <br>mv ./tf_files/flower_photos/daisy/7630520686_e3a61ac763.jpg ./test_images/flower_photos/daisy3.jpg
# <br>mv ./tf_files/flower_photos/dandelion/8475758_4c861ab268_m.jpg ./test_images/flower_photos/dandelion1.jpg
# <br>mv ./tf_files/flower_photos/dandelion/4632761610_768360d425.jpg ./test_images/flower_photos/dandelion2.jpg
# <br>mv ./tf_files/flower_photos/dandelion/15644450971_6a28298454_n.jpg ./test_images/flower_photos/dandelion3.jpg
# <br>mv ./tf_files/flower_photos/sunflowers/27465811_9477c9d044.jpg ./test_images/flower_photos/sunflowers1.jpg
# <br>mv ./tf_files/flower_photos/sunflowers/5020805135_1219d7523d.jpg ./test_images/flower_photos/sunflowers2.jpg
# <br>mv ./tf_files/flower_photos/sunflowers/9056495873_66e351b17c_n.jpg ./test_images/flower_photos/sunflowers3.jpg
# <br>mv ./tf_files/flower_photos/tulips/11746367_d23a35b085_n.jpg ./test_images/flower_photos/tulips1.jpg
# <br>mv ./tf_files/flower_photos/tulips/5674695558_61397a1584.jpg ./test_images/flower_photos/tulips2.jpg
# <br>mv ./tf_files/flower_photos/tulips/13923539227_bdab038dc8.jpg ./test_images/flower_photos/tulips3.jpg
# </div>
# 
# <center>Example Rose Image</center> | <center>Example Tulip Image</center>
# - | - 
# <img src="./images/2a_Rose.jpg" alt="Rose" style="width: 400px;"/> | <img src="./images/2a_Tulip.jpg" alt="Tulip" style="width: 400px;"/>
# 

# <a id='section3'></a><span style="float:right">[Back to Top](#sectionTop)</span>
# # 3. Retraining the Model
# 

# ###### Start up a docker tensorflow container
# <div class="alert alert-block alert-info">
# docker run -it --publish 6006:6006 --volume ${HOME}/tensorflow-for-poets-2/:/tensorflow-for-poets-2 --workdir /tensorflow-for-poets-2 gcr.io/tensorflow/tensorflow:1.1.0 bash
# </div>
# 
# Run the python retraining script help, confirms that the retrain.py is functioning ok
# <div class="alert alert-block alert-info">
# python -m scripts.retrain -h
# </div>
# 
# <br>This script handles 3 things:
# - Download model architecture (only happens once for that model)
# - Make bottlenecks (only happens once for the images being trained)
#     - On Mac laptop, ~4 mins per 1000 images (~300x300 pixel)
#     - On Mac laptop, ~6 mins per 1000 images (~4096x2160 pixel)
# - Retrain model (n training steps executed, 4000 is default)
# 

# # Flowers Data (Google "Tensorflow for Poets" Demo):
# ImageNet can be retrained using the Inception_v3 architecture on the Flowers dataset with the below command
# <br> 15 mins to make bottlenecks the first time
# <br> 3 mins to train 500 steps (claims 89.7% accuracy)
# <br>21 mins to train 4000 steps (claims 91.1% accuracy)
# 

# ```python -m scripts.retrain   --bottleneck_dir=tf_files/bottlenecks   --how_many_training_steps=500   --model_dir=tf_files/models/   --summaries_dir=tf_files/training_summaries/inception_v3   --output_graph=tf_files/output/flowers_500_graph.pb   --output_labels=tf_files/output/flowers_500_labels.txt   --architecture=inception_v3   --image_dir=tf_files/flower_photos```
#   
# <br>Can train longer, say 4000 steps, with:
# 
# ```python -m scripts.retrain   --bottleneck_dir=tf_files/bottlenecks   --how_many_training_steps=4000   --model_dir=tf_files/models/   --summaries_dir=tf_files/training_summaries/inception_v3   --output_graph=tf_files/output/flowers_4000_graph.pb   --output_labels=tf_files/output/flowers_4000_labels.txt   --architecture=inception_v3   --image_dir=tf_files/flower_photos```
# 

# <a id='section4'></a><span style="float:right">[Back to Top](#sectionTop)</span>
# # 4. Validation code for classification model
# 
# As of 9/23/17, there is a bug in label_image.py. Change the following lines:
# 
# ```
# # Change from this:
# input_height = 224
# input_width = 224
# input_mean = 128
# input_std = 128
# input_layer = "input"
# 
# # Change to this:
# input_height = 299
# input_width = 299
# input_mean = 0
# input_std = 255
# input_layer = "Mul"
# ```
# 

# # Flowers Data (Google "Tensorflow for Poets" Demo):
# 

# Model reported 91.1% accuracy after 4000 step training.
# Some actual results below (14/15 correct):
# 
# ```python -m scripts.label_image     --graph=tf_files/output/flowers_4000_graph.pb      --labels=tf_files/output/flowers_4000_labels.txt      --image=test_images/flower_photos/rose1.jpg```
# 
# <center></center> | <center></center> | <center></center>
# - | - | -
# <center>**Rose 1<br>roses 0.999549**<br>tulips 0.000405927<br>sunflowers 3.60402e-05<br>daisy 9.16548e-06<br>dandelion 1.75581e-07</center> | <center>**Rose 2<br>roses 0.968867**<br>tulips 0.027235<br>sunflowers 0.00199636<br>daisy 0.000950742<br>dandelion 0.000950667</center> | <center>**Rose 3<br>roses 0.871816**<br>tulips 0.126929<br>sunflowers 0.00119743<br>dandelion 4.76507e-05<br>daisy 1.0666e-05</center>
# <img src="./images/test/rose1.jpg"/> | <img src="./images/test/rose2.jpg"/> | <img src="./images/test/rose3.jpg"/>
# ||
# ||
# ||
# ||
# <center>**Tulip 1<br>tulips 0.923023**<br>roses 0.0622453<br>dandelion 0.00861938<br>daisy 0.00473263<br>sunflowers 0.00137952</center> | <center>**Tulip 2<br>tulips 0.892067**<br>sunflowers 0.0802674<br>daisy 0.0132293<br>dandelion 0.00992827<br>roses 0.00450842</center> | <center>**Tulip 3<br>tulips 0.882232**<br>dandelion 0.0571973<br>sunflowers 0.0330509<br>roses 0.0170892<br>daisy 0.010431</center>
# <img src="./images/test/tulips1.jpg"/> | <img src="./images/test/tulips2.jpg"/> | <img src="./images/test/tulips3.jpg"/>
# ||
# ||
# ||
# ||
# <center>**Dandelion 1<br>dandelion 0.560596**<br>sunflowers 0.394746<br>daisy 0.0403786<br>tulips 0.00343286<br>roses 0.000846216</center> | <center>**Dandelion 2<br>dandelion 0.999977**<br>sunflowers 9.7611e-06<br>tulips 5.85115e-06<br>roses 4.09581e-06<br>daisy 3.87133e-06</center> | <center>**Dandelion 3**<br><font color="FF0000">**sunflowers 0.544294**</font><br>dandelion 0.399705<br>tulips 0.0297771<br>daisy 0.0259137<br>roses 0.000310016</center>
# <img src="./images/test/dandelion1.jpg"/> | <img src="./images/test/dandelion2.jpg"/> | <img src="./images/test/dandelion3.jpg"/>
# ||
# ||
# ||
# ||
# <center>**Daisy 1<br>daisy 0.985345**<br>dandelion 0.00887047<br>sunflowers 0.00449815<br>tulips 0.00098446<br>roses 0.00030153</center> | <center>**Daisy 2<br>daisy 0.997346**<br>sunflowers 0.00226963<br>dandelion 0.000252017<br>roses 8.0504e-05<br>tulips 5.18256e-05</center> | <center>**Daisy 3<br>daisy 0.995164**<br>sunflowers 0.00406516<br>dandelion 0.000363425<br>tulips 0.000302943<br>roses 0.000104804</center>
# <img src="./images/test/daisy1.jpg"/> | <img src="./images/test/daisy2.jpg"/> | <img src="./images/test/daisy3.jpg"/>
# ||
# ||
# ||
# ||
# <center>**Sunflower 1<br>sunflowers 0.963783**<br>dandelion 0.0171869<br>tulips 0.00798338<br>daisy 0.00797936<br>roses 0.00306722</center> | <center>**Sunflower 2<br>sunflowers 0.971356**<br>dandelion 0.014657<br>daisy 0.0103836<br>tulips 0.00298796<br>roses 0.000615895</center> | <center>**Sunflower 3<br>sunflowers 0.780127**<br>tulips 0.143532<br>dandelion 0.0735062<br>roses 0.00280192<br>daisy 3.27696e-05</center>
# <img src="./images/test/sunflowers1.jpg"/> | <img src="./images/test/sunflowers2.jpg"/> | <img src="./images/test/sunflowers3.jpg"/>
# 




# <a id='sectionTop'></a>
# # Image Classification Demo
# ## ImageNet retrained using Inception_v3 architecture
# 
# [1. Data Prep](#section1)<br>
# [2. Retrain Model](#section2)<br>
# [3. Test It!](#section3)<br>
# 

# <a id='section1'></a><span style="float:right">[Back to Top](#sectionTop)</span>
# # 1. Data Prep
# 
# 1. Capture 10 seconds each of video for 3 students. Same background.
# 1. Capture 4 pictures of each student, 1 with same background, 3 different backgrounds.
# 1. Transfer video/images from phone to laptop<br>
#     Connect Phone to Laptop, Copy files --> DCIM/Camera<br>
#     Copy videos to: Video_Files on desktop --> /Users/epreble/Desktop/Video_Files/<br>
#         Rename videos to student names
#     Copy test images to: Student Test Images shortcut
#         Rename images to student names#
#     Create NAME directories in /Users/epreble/tensorflow-for-poets-2/tf_files/recognize_students/
# 1. Convert 3 videos from mp4 to jpg
# 

import cv2
import os

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, '%d.jpg') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()


# NAME.mp4 - 
video_path = '/Users/epreble/Desktop/Video_Files/NAME1.mp4'
output_image_path = '/Users/epreble/tensorflow-for-poets-2/tf_files/recognize_students/NAME1/'
video_to_frames(video_path, output_image_path)


# NAME.mp4 - 
video_path = '/Users/epreble/Desktop/Video_Files/NAME2.mp4'
output_image_path = '/Users/epreble/tensorflow-for-poets-2/tf_files/recognize_students/NAME2/'
video_to_frames(video_path, output_image_path)


# NAME.mp4 - 
video_path = '/Users/epreble/Desktop/Video_Files/NAME3.mp4'
output_image_path = '/Users/epreble/tensorflow-for-poets-2/tf_files/recognize_students/NAME3/'
video_to_frames(video_path, output_image_path)


# <a id='section2'></a><span style="float:right">[Back to Top](#sectionTop)</span>
# # 2. Retrain Model
# 

# 4 mins to do ~1000 bottlenecks
# <br>2 mins to train 500 steps 
# <br>20 mins to train 4000 steps
# 
# **Start up Docker Tensorflow Container:**
# <br>```docker run -it --publish 6006:6006 --name ep_tensorflow --volume ${HOME}/tensorflow-for-poets-2/:/tensorflow-for-poets-2 --workdir /tensorflow-for-poets-2 gcr.io/tensorflow/tensorflow:1.1.0 bash```
# 
# If need to stop a previous docker container:
# <br>docker ps -a (list containers)
# <br>docker stop container
# <br>docker rm container
# 
# **Run this script to retrain the model with 500 iterations**
# <br>```python -m scripts.retrain   --bottleneck_dir=tf_files/bottlenecks   --how_many_training_steps=500   --model_dir=tf_files/models/   --summaries_dir=tf_files/training_summaries/inception_v3   --output_graph=tf_files/output/students_500_graph.pb   --output_labels=tf_files/output/students_500_labels.txt   --architecture=inception_v3   --image_dir=tf_files/recognize_students```
#   
# **Run this script to retrain the model with 4000 iterations**
# <br>```python -m scripts.retrain   --bottleneck_dir=tf_files/bottlenecks   --how_many_training_steps=4000   --model_dir=tf_files/models/   --summaries_dir=tf_files/training_summaries/inception_v3   --output_graph=tf_files/output/students_4000_graph.pb   --output_labels=tf_files/output/students_4000_labels.txt   --architecture=inception_v3   --image_dir=tf_files/recognize_students```
# 

# <a id='section3'></a><span style="float:right">[Back to Top](#sectionTop)</span>
# # 3. Test it!
# 

# ```python -m scripts.label_image     --graph=tf_files/output/students_500_graph.pb      --labels=tf_files/output/students_500_labels.txt      --image=test_images/recognize_students/NAME1.jpg```
# 
# ```python -m scripts.label_image     --graph=tf_files/output/students_4000_graph.pb      --labels=tf_files/output/students_4000_labels.txt      --image=test_images/recognize_students/NAME1.jpg```
# 
# <center></center> | <center></center> | <center></center> | <center></center>
# - | - | - | -
# <center>**Name**</center>|<center>**Image #**</center>|<center>**500 Iterations**</center>|<center>**4000 Iterations**</center>
# Cross Entropy||0.xxx|0.xxx
# NAME|1||
# |2||
# |3||
# |4||
# NAME|1||
# |2||
# |3||
# |4||
# NAME|1||
# |2||
# |3||
# |4||
# 
# <font color="FF0000">xx<br>(NAME = xx)</font>
# 




