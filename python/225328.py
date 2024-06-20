# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.
# 

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

get_ipython().run_line_magic('matplotlib', 'inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.
# 

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights
# 

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


print(COCO_MODEL_PATH)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)
# 

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection
# 

# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

IMAGE_DIR = '/Users/NYX/Desktop/Project/mapillary-vistas-dataset_public_v1/testing/images'
IMAGE_DIR = '/Users/NYX/Documents/FamilyAlbum/171225Peru_Bolivia'
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR,  file_names[25]))
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])





# # Mask R-CNN - Train on Mapillary Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 
# 

import os
import coco
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import json
from PIL import Image
from config import Config
import utils
import model as modellib
import visualize
from model import log

get_ipython().run_line_magic('matplotlib', 'inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# local path to mapillary dataset
DATASET_DIR = os.path.join(ROOT_DIR, "mapillary_dataset")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class MapillaryConfig(coco.CocoConfig):
    """Configuration for training on the mapillary dataset.
    Derives from the base Config class and overrides values specific
    to the mapillary dataset.
    """
    # Give the configuration a recognizable name
    NAME = "train_4096"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    # this MUST be explicitly defined, or will run into index out of bound error
    NUM_CLASSES = 1 + 11  # background + 11 objects

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0005
    
config = MapillaryConfig()
config.display()


class MapillaryDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    
    DEBUG = False
    CLASS_MAP = {}
    CLASSES = ["Bird", "Person", "Bicyclist", "Motorcyclist", "Bench",      "Car", "Person", "Fire Hydrant", "Traffic Light", "Bus", "Motorcycle", "Truck"]
    
    # local path to image folder, choose 'dev', 'training', or 'testing'
    SUBSET_DIR = ""

    # local path to images inside development folder
    IMG_DIR = ""

    # local path to instance annotations inside development folder
    INS_DIR = ""


    def load_mapillary(self, dataset_dir, subset, class_ids=None,
                  class_map=None):
        
        self.SUBSET_DIR = os.path.join(dataset_dir, subset)
        self.IMG_DIR = os.path.join(self.SUBSET_DIR, 'images')
        self.INS_DIR = os.path.join(self.SUBSET_DIR, 'instances')
        
        # load classes, start with id = 1 to account for background "BG"
        class_id = 1
        for label_id, label in enumerate(class_ids):
            if label["instances"] == True and label["readable"] in self.CLASSES:
                self.CLASS_MAP[label_id] = class_id
                
                if (self.DEBUG):
                    print("{}: Class {} {} added".format(label_id, class_id, label["readable"]))
                    
                self.add_class("mapillary", class_id, label["readable"])
                class_id = class_id + 1
                
        # add images 
        file_names = next(os.walk(self.IMG_DIR))[2]
        for i in range(len(file_names)):
            if file_names[i] != '.DS_Store':
                image_path = os.path.join(self.IMG_DIR, file_names[i])
                base_image = Image.open(image_path)
                w, h = base_image.size

                if (self.DEBUG):
                    print("Image {} {} x {} added".format(file_names[i], w, h))

                self.add_image("mapillary", image_id = i,
                              path = file_names[i],
                               width = w,
                               height = h
                              )

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        This function loads the image from a file.
        """
        info = self.image_info[image_id]
        img_path = os.path.join(self.IMG_DIR, info["path"])
        image = Image.open(img_path)
        image_array = np.array(image)
        return image_array

    def image_reference(self, image_id):
        """Return the local directory path of the image."""
        info = self.image_info[image_id]
        img_path = os.path.join(self.IMG_DIR, info["path"])
        return img_path

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        instance_path = os.path.join(self.INS_DIR, info["path"])
        instance_image = Image.open(instance_path.rsplit(".", 1)[0] + ".png")
        
        # convert labeled data to numpy arrays for better handling
        instance_array = np.array(instance_image, dtype=np.uint16)

        instances = np.unique(instance_array)
        instaces_count = instances.shape[0]
        
        label_ids = instances // 256
        label_id_count = np.unique(label_ids).shape[0]
        
        if (self.DEBUG):
            print("There are {} instances, {} classes labelled instances in the image {}."                  .format(instaces_count, label_id_count, info["path"]))
            
        mask = np.zeros([instance_array.shape[0], instance_array.shape[1], instaces_count], dtype=np.uint8)
        mask_count = 0
        loaded_class_ids = []
        for instance in instances:
            label_id = instance//256
            if (label_id in self.CLASS_MAP):
                m = np.zeros((instance_array.shape[0], instance_array.shape[1]), dtype=np.uint8)
                m[instance_array == instance] = 1
                m_size = np.count_nonzero(m == 1)
                
                # only load mask greater than threshold size, 
                # otherwise bounding box with area zero causes program to crash
                if m_size > 4096:
                    mask[:, :, mask_count] = m
                    loaded_class_ids.append(self.CLASS_MAP[label_id])
                    mask_count = mask_count + 1
                    if (self.DEBUG):
                        print("Non-zero: {}".format(m_size))
                        print("Mask {} created for instance {} of class {} {}"                              .format(mask_count, instance, self.CLASS_MAP[label_id],                                       self.class_names[self.CLASS_MAP[label_id]]))
        mask = mask[:, :, 0:mask_count]
        return mask, np.array(loaded_class_ids)


# read in config file
with open(os.path.join(DATASET_DIR, 'config.json')) as config_file:
    class_config = json.load(config_file)
# in this example we are only interested in the labels
labels = class_config['labels']
        
# Training dataset
dataset_train = MapillaryDataset()
dataset_train.load_mapillary(DATASET_DIR, "train_107", class_ids = labels)
dataset_train.prepare()

# Validation dataset
dataset_val = MapillaryDataset()
dataset_val.load_mapillary(DATASET_DIR, "dev_9", class_ids = labels)
dataset_val.prepare()

print("mapping: ", class_config["mapping"])
print("version: ", class_config["version"])
print("folder_structure:", class_config["folder_structure"])
print("There are {} classes in the config file".format(len(labels)))
print("There are {} classes in the model".format(len(dataset_train.class_names)))
for i in range(len(dataset_train.class_names)):
    print("    Class {}: {}".format(i, dataset_train.class_names[i]))


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
    
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=2, 
            layers='heads')





import os
import json
from PIL import Image
import numpy as np
import random
from shutil import copy2, move


ROOT_DIR = os.getcwd()
DATASET_DIR = os.path.join(ROOT_DIR, "mapillary_dataset")
ORIGINAL_20000_IMG = os.path.join(DATASET_DIR, "original_20000", "images")
ORIGINAL_20000_INS = os.path.join(DATASET_DIR, "original_20000", "instances")

DS_STORE = '.DS_Store'

intersection_class_ids = [0, 19, 20, 21, 33, 38, 48, 54, 55, 57, 61]
instance_class_ids = [0, 1, 8, 19, 20, 21, 22, 23, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62]


# ### Helper Functions
# 

# write a list of files to .txt
def Write_TXT(file_names, dst, txt_name):
    if not os.path.exists(dst):
            os.makedirs(dst)
    with open(os.path.join(dst, txt_name), 'w') as the_file:
        for file_name in file_names:
            the_file.write(file_name.rsplit(".", 1)[0])
            the_file.write('\n')


def Read_TXT(txt_path):
    with open(txt_path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


# Python code t get difference of two lists
# Using set()
def Diff(li1, li2):
    return (list(set(li1) - set(li2)))


# Copy a list of files to a desired directory
def Copy_Files(dataset_name, subset_name, file_names):
    for file_name in file_names:
        src = os.path.join(ORIGINAL_20000_IMG, file_name + ".jpg")
        dst = os.path.join(DATASET_DIR, dataset_name, subset_name, "images")
        if not os.path.exists(dst):
            os.makedirs(dst)
        copy2(src, dst)

        ins_name = file_name.rsplit(".", 1)[0] + ".png"
        src = os.path.join(ORIGINAL_20000_INS, ins_name)
        dst = os.path.join(DATASET_DIR, dataset_name, subset_name, "instances")
        if not os.path.exists(dst):
            os.makedirs(dst)
        copy2(src, dst)


def Read_Dataset(txt_path):
    if os.path.exists(txt_path):
        all_files = Read_TXT(txt_path)
    else:
        IMG_DIR = os.path.join(txt_path.rsplit(".", 1)[0], "images")
        print(IMG_DIR)
        all_files = next(os.walk(IMG_DIR))[2]
        if DS_STORE in all_files:
            all_files.remove(DS_STORE)
        Write_TXT(all_files, IMG_DIR, txt_path)
    return all_files


# Select desired number of examples whose ground truth masks are non-trivial (defined by threshold)
# this function was modified from dataset_clean.ipynb
def Select_Examples(available_files, num_files_needed, threshold, class_ids):
    accepted = []
    rejected = []
    idx = 0
    while len(accepted) < num_files_needed:
        if idx >= len(available_files):
            print("not enough avaialble files!")
            break
            
        file_name = available_files[idx]
        if file_name != '.DS_Store':
            IMG_PATH = os.path.join(ORIGINAL_20000_IMG, file_name)
            ins_name = file_name.rsplit(".", 1)[0] + ".png"
            INS_PATH = os.path.join(ORIGINAL_20000_INS, ins_name)
            instance_image = Image.open(INS_PATH)
            
            # convert labeled data to numpy arrays for better handling
            instance_array = np.array(instance_image, dtype=np.uint16)

            instances = np.unique(instance_array)
            instaces_count = instances.shape[0]

            label_ids = instances // 256
            label_id_count = np.unique(label_ids).shape[0]

            mask_count = 0
            for instance in instances:
                label_id = instance // 256
                if label_id in class_ids:
                    m = np.zeros((instance_array.shape[0], instance_array.shape[1]), dtype=np.uint8)
                    m[instance_array == instance] = 1
                    m_size = np.count_nonzero(m == 1)

                    # only load mask greater than threshold size, 
                    # otherwise bounding box with area zero causes program to crash
                    if m_size > threshold:
                        mask_count = mask_count + 1
            if mask_count == 0:
                rejected.append(file_name)
            else:
                accepted.append(file_name)
        idx = idx + 1
        print('Accepted {}/{}, rejected {}\r'.format(len(accepted), num_files_needed, len(rejected)), end='', )
        

        with open(os.path.join(DATASET_DIR, "progress.txt"), 'w') as the_file:
            the_file.write('Accepted {}/{}, rejected {}\r'.format(len(accepted), num_files_needed, len(rejected)))
            the_file.write('\n')
            
    return accepted, rejected


def Build_Dataset(available_files, dataset_name, subset_name, threshold, size, rebuild = False, class_ids = intersection_class_ids):
    accepted = []
    rejected = []

    subset_dir = os.path.join(DATASET_DIR, dataset_name, subset_name)
    accepted_txt = os.path.join(subset_dir, "accepted.txt")
    rejected_txt = os.path.join(subset_dir, "rejected.txt")

    if rebuild:
        accepted, rejected = Select_Examples(available_files, size, threshold, class_ids)

        Write_TXT(accepted, subset_dir, "accepted.txt")
        Write_TXT(rejected, subset_dir, "rejected.txt")

        Copy_Files(subset_dir, "accepted", accepted)
        Copy_Files(subset_dir, "rejected", rejected)
    else:
        accepted = Read_Dataset(accepted_txt)
        rejected = Read_Dataset(rejected_txt)
    
    used = accepted + rejected
    available_files = Diff(available_files, used)

    print('{} accepted {} and rejected {}'.format(dataset_name, len(accepted), len(rejected)))
    print('{} images still available'.format(len(available_files)))
    
    return available_files, accepted, rejected


# ### Examine the original 20k images
# 

txt_20k = os.path.join(DATASET_DIR, "original_20000.txt")
all_files = Read_Dataset(txt_20k)
print('{} images were found in the mapillary dataset'.format(len(all_files)))


# ### AWS run 1 (mask threshold 64^2)
# 

AWS_RUN_1 = os.path.join(DATASET_DIR, "AWS_run_1")


txt_path = os.path.join(DATASET_DIR, "AWS_run_1", "train_4096.txt")
train_file_names = Read_Dataset(txt_path)

txt_path = os.path.join(DATASET_DIR, "AWS_run_1", "dev_512.txt")
dev_file_names = Read_Dataset(txt_path)

used_files = list(set(train_file_names + dev_file_names))
    
print('{} images were used in AWS run 1 (train + dev)'.format(len(used_files)))


# take difference between two lists
available_files = Diff(all_files, used_files)
print('{} images still available'.format(len(available_files)))


# ### Make it reproducible
# #### ref: https://cs230-stanford.github.io/train-dev-test-split.html
# 

# make sure that the filenames have a fixed order before shuffling
available_files.sort()  

# fix the random seed
random.seed(0)

# shuffles the ordering of filenames (deterministic given the chosen seed)
random.shuffle(available_files) 


# ### AWS Run 2 (mask threshold 32^2)¶
# 

available_files, AWS_run_2_train, AWS_run_2_train_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_2", 
                                                                           subset_name = "train_4096",
                                                                           threshold = 32 * 32, 
                                                                           size = 4096, 
                                                                           rebuild = False)


available_files, AWS_run_2_dev, AWS_run_2_dev_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_2", 
                                                                           subset_name = "dev_512",
                                                                           threshold = 32 * 32, 
                                                                           size = 512, 
                                                                           rebuild = False)


# ### AWS Run 3 (Full dataset, mask threshold 32^2)
# 

txt_20k = os.path.join(DATASET_DIR, "original_20000.txt")
all_files = Read_Dataset(txt_20k)
print('{} images were found in the mapillary dataset'.format(len(all_files)))


# make sure that the filenames have a fixed order before shuffling
all_files.sort()  

# fix the random seed
random.seed(230)

# shuffles the ordering of filenames (deterministic given the chosen seed)
random.shuffle(all_files) 


available_files, AWS_run_3_dev, AWS_run_3_dev_rejected = Build_Dataset(available_files = all_files,
                                                                           dataset_name = "AWS_run_3", 
                                                                           subset_name = "dev_1024",
                                                                           threshold = 32 * 32, 
                                                                           size = 1024, 
                                                                           rebuild = False)


available_files, AWS_run_3_test, AWS_run_3_test_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_3", 
                                                                           subset_name = "test_1024",
                                                                           threshold = 32 * 32, 
                                                                           size = 1024, 
                                                                           rebuild = False)


available_files, AWS_run_3_train, AWS_run_3_train_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_3", 
                                                                           subset_name = "train_16384",
                                                                           threshold = 32 * 32, 
                                                                           size = 16384, 
                                                                           rebuild = False)


# ### AWS Run 4 (Full dataset, 38 classes, mask threshold 32^2)
# 

txt_20k = os.path.join(DATASET_DIR, "original_20000.txt")
all_files = Read_Dataset(txt_20k)
print('{} images were found in the mapillary dataset'.format(len(all_files)))


# make sure that the filenames have a fixed order before shuffling
all_files.sort()  

# fix the random seed
random.seed(230)

# shuffles the ordering of filenames (deterministic given the chosen seed)
random.shuffle(all_files) 


available_files, AWS_run_4_dev, AWS_run_4_dev_rejected = Build_Dataset(available_files = all_files,
                                                                           dataset_name = "AWS_run_4", 
                                                                           subset_name = "dev_1024",
                                                                           threshold = 32 * 32, 
                                                                           size = 1024, 
                                                                           rebuild = False,
                                                                           class_ids = instance_class_ids)


available_files, AWS_run_4_test, AWS_run_4_test_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_4", 
                                                                           subset_name = "test_1024",
                                                                           threshold = 32 * 32, 
                                                                           size = 1024, 
                                                                           rebuild = False,
                                                                           class_ids = instance_class_ids)


available_files, AWS_run_4_train, AWS_run_4_train_rejected = Build_Dataset(available_files = available_files,
                                                                           dataset_name = "AWS_run_4", 
                                                                           subset_name = "train_16384",
                                                                           threshold = 32 * 32, 
                                                                           size = 16384, 
                                                                           rebuild = False,
                                                                           class_ids = instance_class_ids)


# ### Split train set into 8 parts of 2048 images to upload individually
# 

dataset_name = "AWS_run_4"


# copy files from original_20000 if messed up
# Copy_Files(os.path.join(DATASET_DIR, dataset_name, "train_16384"), "accepted", AWS_run_4_train)


# divide images into 8 folders
files_to_move = AWS_run_4_train
part_size = 2048
directory = os.path.join(DATASET_DIR, dataset_name, "train_16384", "accepted", "images")
num_parts = int(len(files_to_move) / part_size)

num_files = next(os.walk(directory))[1]
if (len(num_files) == len(files_to_move)):
    for i in range(num_parts):
        part = files_to_move[:part_size]
        part_dir = os.path.join(directory, "part_" + str(i))
        if not os.path.exists(part_dir):
            os.makedirs(part_dir)
        for file in part:
            src = os.path.join(directory, file + '.jpg')
            move(src, part_dir)
        files_to_move = Diff(files_to_move, part)


# check the number of images in each folder
directory = os.path.join(DATASET_DIR, dataset_name, "train_16384", "accepted", "images")
folders = next(os.walk(directory))[1]
folders.sort()  
for folder in folders:
    part_dir = os.path.join(directory, folder)
    files = next(os.walk(part_dir))[2]
    if DS_STORE in files:
        files.remove(DS_STORE)
    print("num_files in {}: {}".format(folder, len(files)))


# # generate datasets sequentially withut rejecting anything (not recommended)
# TRAIN_BATCH_SIZE = 4096
# DEV_BATCH_SIZE = 512
# TEST_SET_SIZE = len(available_files) - (TRAIN_BATCH_SIZE + DEV_BATCH_SIZE) * 3

# start = 0
# AWS_run_2_train = available_files[:TRAIN_BATCH_SIZE]
# AWS_run_2_dev   = available_files[TRAIN_BATCH_SIZE : TRAIN_BATCH_SIZE + DEV_BATCH_SIZE]

# AWS_run_3_train = available_files[TRAIN_BATCH_SIZE + DEV_BATCH_SIZE : TRAIN_BATCH_SIZE * 2 + DEV_BATCH_SIZE]
# AWS_run_3_dev   = available_files[TRAIN_BATCH_SIZE * 2 + DEV_BATCH_SIZE : (TRAIN_BATCH_SIZE + DEV_BATCH_SIZE) * 2]


# AWS_run_4_train = available_files[(TRAIN_BATCH_SIZE + DEV_BATCH_SIZE) * 2 : TRAIN_BATCH_SIZE * 3 + DEV_BATCH_SIZE * 2]
# AWS_run_4_dev   = available_files[TRAIN_BATCH_SIZE * 3 + DEV_BATCH_SIZE *2 : (TRAIN_BATCH_SIZE + DEV_BATCH_SIZE) * 3]

# assert(len(AWS_run_2_train) == 4096)
# assert(len(AWS_run_3_train) == 4096)
# assert(len(AWS_run_4_train) == 4096)

# assert(len(AWS_run_2_dev) == 512)
# assert(len(AWS_run_3_dev) == 512)
# assert(len(AWS_run_4_dev) == 512)

# test_set = available_files[(TRAIN_BATCH_SIZE + DEV_BATCH_SIZE) * 3 :]
# print("{} images in the test set".format(len(test_set)))





from __future__ import print_function
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array


# a nice example
key = '_7zhntDU5r1EmkFSuzKxaQ'

# read in config file
with open('config.json') as config_file:
    config = json.load(config_file)
# in this example we are only interested in the labels
labels = config['labels']

print("mapping: ", config["mapping"])
print("version: ", config["version"])
print("folder_structure:", config["folder_structure"])
print("There are {} labels in the config file".format(len(labels)))


# We provide pixel-wise labels based on polygon annotations for 66 object classes, where 37 are annotated in an instance-specific manner (i.e. individual instances are labeled separately). 
# 

for label_id, label in enumerate(labels):
    print("{:>30} ({:2d}): {:<50} has instances: {}".format(label["readable"], label_id, label["name"], label["instances"]))


for label_id, label in enumerate(labels):
    if(label["instances"]):
             print(label["readable"])


# set up paths for every image
image_path = "training/images/{}.jpg".format(key)
label_path = "training/labels/{}.png".format(key)
instance_path = "training/instances/{}.png".format(key)


# load images
base_image = Image.open(image_path)
label_image = Image.open(label_path)
instance_image = Image.open(instance_path)


# convert labeled data to numpy arrays for better handling
label_array = np.array(label_image)

# for visualization, we apply the colors stored in the config
colored_label_array = apply_color_map(label_array, labels)


r = 1050
c = 3600
print('Pixel at [{}, {}] is: {}'.format(r, c, labels[label_array[r, c]]['readable']))


# convert labeled data to numpy arrays for better handling
instance_array = np.array(instance_image, dtype=np.uint16)

# now we split the instance_array into labels and instance ids
instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
colored_instance_label_array = apply_color_map(instance_label_array, labels)

# instance ids
instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)


a = np.array([[1, 2, 3], [3, 2, 1], [4, 3, 2]])
m = np.zeros((3, 3))
m[a == 1] = 1
print(m)


instances = np.unique(instance_array)
class_ids = instances // 256

instaces_count = instances.shape[0]
classes_id_count = np.unique(class_ids).shape[0]

mask = np.zeros([instance_array.shape[0], instance_array.shape[1], instaces_count], dtype=np.uint8)
print("There are {} masks, {} classes in this image".format(instaces_count, classes_id_count))


for i in range(instaces_count):
    m = np.zeros((instance_array.shape[0], instance_array.shape[1]))
    m[instance_array == instances[i]] = 1
    mask[:, :, i] = m
    print('New mask {} created: instance {} of class {}'.format(i, instances[i], labels[class_ids[i]]["readable"]))


r = 2700
c = 1000
print('Pixel at [{}, {}] is labelled: {}, instance: {}'.format(r, c, labels[instance_label_array[r, c]]['readable'], instance_ids_array[r, c]))


print(class_ids)
# plot a mask
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,15))

n = 52
ins = instances[n]
print('Mask {}: instance{} of class {}'.format(n, ins, labels[ins//256]["readable"]))

ax.imshow(mask[:, :, n])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title(labels[ins//256]["readable"])





# plot the result
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,15))

ax[0][0].imshow(base_image)
ax[0][0].get_xaxis().set_visible(False)
ax[0][0].get_yaxis().set_visible(False)
ax[0][0].set_title("Base image")

ax[0][1].imshow(colored_label_array)
ax[0][1].get_xaxis().set_visible(False)
ax[0][1].get_yaxis().set_visible(False)
ax[0][1].set_title("Labels")

ax[1][0].imshow(instance_ids_array)
ax[1][0].get_xaxis().set_visible(False)
ax[1][0].get_yaxis().set_visible(False)
ax[1][0].set_title("Instance IDs")

ax[1][1].imshow(colored_instance_label_array)
ax[1][1].get_xaxis().set_visible(False)
ax[1][1].get_yaxis().set_visible(False)
ax[1][1].set_title("Labels from instance file (identical to labels above)")

fig.savefig('MVD_plot.png')





