from __future__ import division, absolute_import, print_function
import os, sys, time, re, json
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import StringIO
import json
import cv2

imread = plt.imread
def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    img = Image.open(StringIO.StringIO(res))
    return np.array(img)

def read_npy(res):
    return np.load(res)

# Matches a color from the object_mask and then returns that region color
def match_color(object_mask, target_color, tolerance=3):
    match_region = np.ones(object_mask.shape[0:2], dtype=bool)
    for c in range(3): # r,g,b
        min_val = target_color[c] - tolerance
        max_val = target_color[c] + tolerance
        channel_region = (object_mask[:,:,c] >= min_val) & (object_mask[:,:,c] <= max_val)
        match_region &= channel_region

    if match_region.sum() != 0:
        return match_region
    else:
        return None
    
# Swap colors method
def swap_color(imgarray, source, dest):
    matched_color = match_color(imgarray, [source.R, source.G, source.B])
    imgarray[:,:,:3][matched_color] = [dest.R, dest.G, dest.B]
    return np.array(imgarray)


# Connect to the game
# ===================
# Load unrealcv python client, do :code:`pip install unrealcv` first.
# 
# 

from unrealcv import client
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
    sys.exit(-1)


# Make sure the connection works well

res = client.request('vget /unrealcv/status')
# The image resolution and port is configured in the config file.
print(res)


# Get objects
# ======================
# Write a json file with the object and their corresponding classes.
# 
# 

scene_objects = client.request('vget /objects').split(' ')
print('Number of objects in this scene:', len(scene_objects))

if '257' in scene_objects:
    scene_objects.remove('257')

obj_id_to_class = {}
for obj_id in scene_objects:
    obj_id_parts = obj_id.split('_')
    class_name = obj_id_parts[0]    
    obj_id_to_class[obj_id] = class_name

# Write JSON file
with open('neighborhood_object_ids.json', 'w') as outfile:
    json.dump(obj_id_to_class, outfile)


# Get object colors
# ======================
# First we create the color object class
# 

class Color(object):
    ''' A utility class to parse color value '''
    regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
    def __init__(self, color_str):
        self.color_str = color_str
        match = self.regexp.match(color_str)
        (self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1,5)]

    def __repr__(self):
        return self.color_str


# Then, we either load from json
# 

id2color = {}
with open('id2color.json') as data_file:
    data = json.load(data_file)

for obj_id in data.keys():
    color_map = data[obj_id]
    color_str = '(R=' + str(color_map['R']) + ',G=' +                 str(color_map['G']) + ',B=' + str(color_map['B']) +                 ',A=' + str(color_map['A']) + ')'
    color = Color(color_str)
    id2color[obj_id] = color


# or We load from the scene
# 

id2color = {} # Map from object id to the labeling color
for i, obj_id in enumerate(scene_objects):
    color = Color(client.request('vget /object/%s/color' % obj_id))
    id2color[obj_id] = color
    print('%d. %s : %s' % (i, obj_id, str(color)))


# Write to JSON if loaded from scene
# 

# Convert to serializable json dictionary
serializable_map = {}
for color_id in id2color.keys():
    curr_color = id2color[color_id]
    color_map = {}
    color_map['R'] = curr_color.R
    color_map['G'] = curr_color.G
    color_map['B'] = curr_color.B
    color_map['A'] = curr_color.A
    serializable_map[color_id] = color_map

# Write to JSON
with open('id2color.json', 'w') as outfile:
    json.dump(serializable_map, outfile)


# Map classes to lists of objects
classes = {}

for obj_id in obj_id_to_class.keys():
    
    curr_class = obj_id_to_class[obj_id]
    if curr_class not in classes:
        classes[curr_class] = []
    
    classes[curr_class].append(obj_id)

# Write classes to json
with open('neighborhood_classes.json', 'w') as outfile:
    json.dump(classes, outfile) 
    


# Normalize using built in API
counter = 0
for curr_class in classes.keys():
    
    
    object_list = classes[curr_class]
    curr_color = id2color[object_list[0]]
    
    for obj_id in object_list:
        
        client.request('vset /object/' + obj_id + '/color ' +                        str(curr_color.R) + ' ' + str(curr_color.G) +                        ' ' + str(curr_color.B))
        
        print(str(counter) + '. vset /object/' + obj_id + '/color ' +                        str(curr_color.R) + ' ' + str(curr_color.G) +                        ' ' + str(curr_color.B))
    
        counter += 1
    


# Begin Data Collection (Without Normalization)
# ======
# 

with open('finalTopClassesToColor.json', 'r') as infile:
    top20tocolor = json.load(infile)

top20toint = {}
for cnt, cls in enumerate(top20tocolor.keys()):
    top20toint[cls] = cnt + 1

top20toint['Oak'] = top20toint['Fir']
top20toint['Birch'] = top20toint['Fir']
top20toint['Tree'] = top20toint['Fir']

with open('topClassesToInt.json', 'w') as outfile:
    json.dump(top20toint, outfile)

for batch in range(1,1001):

    # Get random location
    z = 300
    x = random.randint(-5500, 5500)
    y = random.randint(-5500, 5500)
    
    # Get random yaw
    yaw = random.randint(0,360)

    # Coordinates x, y, z
    client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                    ' ' + str(z)) 

    # Get 10 shots in a series
    angles = []
    a = 0
    while len(angles) < 20:
        angles.append(a)
        a -= 3

    # Increment height sequentially
    heights = []
    height = 300
    while len(heights) < 20:
        heights.append(height)
        height += 50

    for i,angle in enumerate(angles):
        
        print("Round: " + str(batch) + " , Image: " + str(i))
        
        # x, y, z
        client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                        ' ' + str(heights[i])) 

        # Pitch, yaw, roll
        client.request('vset /camera/0/rotation ' + str(angle) + ' ' + str(yaw) + ' 0')

        # Get Ground Truth
        res = client.request('vget /camera/0/object_mask png')
        object_mask = read_png(res)
        segmentation_image = Image.fromarray(object_mask)
        
        directory = 'D:/segmentation_data/batch7/round' + str(batch) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
                    
        segmentation_image.save('D:/segmentation_data/batch7/round' + str(batch) + '/seg' +                             str(i) + '.png')
        
        print('D:/segmentation_data/batch7/round' + str(batch) + '/seg' +                             str(i) + '.png')
        
        res = client.request('vget /camera/0/lit png')
        normal = read_png(res)
        normal = Image.fromarray(normal)
        
       
        normal.save('D:/segmentation_data/batch7/round' + str(batch) + '/pic' +                             str(i) + '.png')
    
        
        print("Images written. ")


# Begin Data Collection (With Normalization)
# ======
# 

for batch in range(1,5):

    # Get random location
    z = 300
    x = random.randint(-5000, 5000)
    y = random.randint(-5000, 5000)

    # Coordinates x, y, z
    client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                    ' ' + str(z)) 

    # Get 10 shots in a series
    angles = []
    a = 0
    while len(angles) < 20:
        angles.append(a)
        a -= 3

    # Increment height sequentially
    heights = []
    height = 300
    while len(heights) < 20:
        heights.append(height)
        height += 50

    for i,angle in enumerate(angles):
        
        print("Batch: " + str(batch) + " , Image: " + str(i))
        
        # x, y, z
        client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                        ' ' + str(heights[i])) 

        # Pitch, yaw, roll
        client.request('vset /camera/0/rotation ' + str(angle) + ' 0 0')

        """
        res = client.request('vget /camera/0/object_mask ./batch/round' + \
                             str(batch) + '/lit' + str(i) + '.png')
        
        print('The image is saved to %s' % res)
        """
        
        # Get Ground Truth
        res = client.request('vget /camera/0/object_mask png')
        object_mask = read_png(res)
        
        # Get all classes in image
        print("Getting all object_ids in image....")
        id2mask = {}
        for obj_id in scene_objects:
            color = id2color[obj_id]
            mask = match_color(object_mask, [color.R, color.G, color.B], tolerance = 3)
            if mask is not None:
                id2mask[obj_id] = mask
        
        print("Got all object_ids in image.")

        """
        Group classes together
        """ 
        class_groups = {}

        # Go through the matched objects
        for idmask in id2mask.keys():

            # Get the class from the neighborhood classes object
            curr_class = obj_id_to_class[idmask]

            # If the class is not in the class_groups map, add it
            if curr_class not in class_groups:
                class_groups[curr_class] = [] 

            # Add the idmask to it's corresponding class
            class_groups[curr_class].append(idmask)


        # Normalize by class
        for cls in class_groups.keys():
            class_base = class_groups[cls][0]
            class_color = id2color[class_base]
            for class_id in class_groups[cls]:
                current_color = id2color[class_id]
                object_mask = swap_color(object_mask, current_color, class_color) 
        
        # Write to file
        print(object_mask.shape)
        normalized_img = Image.fromarray(object_mask)

        directory = './batch/round' + str(batch) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        normalized_img.save('./batch/round' + str(batch) + '/seg' +                             str(i) + '.png')
        
        res = client.request('vget /camera/0/lit png')
        normal = read_png(res)
        print(normal.shape)
        normal = Image.fromarray(normal)
        
        normal.save('./batch/round' + str(batch) + '/pic' +                             str(i) + '.png')
        
        print("Images written. ")





from __future__ import division, absolute_import, print_function
import os, sys, time, re, json
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import StringIO
import json
import cv2

imread = plt.imread
def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    img = Image.open(StringIO.StringIO(res))
    return np.array(img)

def read_npy(res):
    return np.load(res)

# Matches a color from the object_mask and then returns that region color
def match_color(object_mask, target_color, tolerance=3):
    match_region = np.ones(object_mask.shape[0:2], dtype=bool)
    for c in range(3): # r,g,b
        min_val = target_color[c] - tolerance
        max_val = target_color[c] + tolerance
        channel_region = (object_mask[:,:,c] >= min_val) & (object_mask[:,:,c] <= max_val)
        match_region &= channel_region

    if match_region.sum() != 0:
        return match_region
    else:
        return None
    
# Swap colors method
def swap_color(imgarray, source, dest):
    matched_color = match_color(imgarray, [source.R, source.G, source.B])
    imgarray[:,:,:3][matched_color] = [dest.R, dest.G, dest.B]
    return np.array(imgarray)


# Connect to the game
# ===================
# Load unrealcv python client, do :code:`pip install unrealcv` first.
# 
# 

from unrealcv import client
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
    sys.exit(-1)


# Make sure the connection works well

res = client.request('vget /unrealcv/status')
# The image resolution and port is configured in the config file.
print(res)


# Get objects
# ======================
# Write a json file with the object and their corresponding classes.
# 
# 

scene_objects = client.request('vget /objects').split(' ')
print('Number of objects in this scene:', len(scene_objects))

if '257' in scene_objects:
    scene_objects.remove('257')

obj_id_to_class = {}
for obj_id in scene_objects:
    obj_id_parts = obj_id.split('_')
    class_name = obj_id_parts[0]    
    obj_id_to_class[obj_id] = class_name

# Write JSON file
with open('neighborhood_object_ids.json', 'w') as outfile:
    json.dump(obj_id_to_class, outfile)


# Get object colors
# ======================
# First we create the color object class
# 

class Color(object):
    ''' A utility class to parse color value '''
    regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
    def __init__(self, color_str):
        self.color_str = color_str
        match = self.regexp.match(color_str)
        (self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1,5)]

    def __repr__(self):
        return self.color_str


# Then, we either load from json
# 

id2color = {}
with open('id2color.json') as data_file:
    data = json.load(data_file)

for obj_id in data.keys():
    color_map = data[obj_id]
    color_str = '(R=' + str(color_map['R']) + ',G=' +                 str(color_map['G']) + ',B=' + str(color_map['B']) +                 ',A=' + str(color_map['A']) + ')'
    color = Color(color_str)
    id2color[obj_id] = color


# or We load from the scene
# 

id2color = {} # Map from object id to the labeling color
for i, obj_id in enumerate(scene_objects):
    color = Color(client.request('vget /object/%s/color' % obj_id))
    id2color[obj_id] = color
    print('%d. %s : %s' % (i, obj_id, str(color)))


# Write to JSON if loaded from scene
# 

# Convert to serializable json dictionary
serializable_map = {}
for color_id in id2color.keys():
    curr_color = id2color[color_id]
    color_map = {}
    color_map['R'] = curr_color.R
    color_map['G'] = curr_color.G
    color_map['B'] = curr_color.B
    color_map['A'] = curr_color.A
    serializable_map[color_id] = color_map

# Write to JSON
with open('id2color.json', 'w') as outfile:
    json.dump(serializable_map, outfile)


# Map classes to lists of objects
classes = {}

for obj_id in obj_id_to_class.keys():
    
    curr_class = obj_id_to_class[obj_id]
    if curr_class not in classes:
        classes[curr_class] = []
    
    classes[curr_class].append(obj_id)

# Write classes to json
with open('neighborhood_classes.json', 'w') as outfile:
    json.dump(classes, outfile) 
    


# Normalize using built in API
counter = 0
for curr_class in classes.keys():
    
    
    object_list = classes[curr_class]
    curr_color = id2color[object_list[0]]
    
    for obj_id in object_list:
        
        client.request('vset /object/' + obj_id + '/color ' +                        str(curr_color.R) + ' ' + str(curr_color.G) +                        ' ' + str(curr_color.B))
        
        print(str(counter) + '. vset /object/' + obj_id + '/color ' +                        str(curr_color.R) + ' ' + str(curr_color.G) +                        ' ' + str(curr_color.B))
    
        counter += 1
    


# Begin Data Collection (Without Normalization)
# ======
# 

with open('finalTopClassesToColor.json', 'r') as infile:
    top20tocolor = json.load(infile)

top20toint = {}
for cnt, cls in enumerate(top20tocolor.keys()):
    top20toint[cls] = cnt + 1

top20toint['Oak'] = top20toint['Fir']
top20toint['Birch'] = top20toint['Fir']
top20toint['Tree'] = top20toint['Fir']

with open('topClassesToInt.json', 'w') as outfile:
    json.dump(top20toint, outfile)

for batch in range(1,301):

    # Get random location
    z = 300
    x = random.randint(-5500, 5500)
    y = random.randint(-5500, 5500)

    # Coordinates x, y, z
    client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                    ' ' + str(z)) 

    # Get 10 shots in a series
    angles = []
    a = 0
    while len(angles) < 20:
        angles.append(a)
        a -= 3

    # Increment height sequentially
    heights = []
    height = 300
    while len(heights) < 20:
        heights.append(height)
        height += 50

    for i,angle in enumerate(angles):
        
        print("Batch: " + str(batch) + " , Image: " + str(i))
        
        # x, y, z
        client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                        ' ' + str(heights[i])) 

        # Pitch, yaw, roll
        client.request('vset /camera/0/rotation ' + str(angle) + ' 0 0')

        # Get Ground Truth
        res = client.request('vget /camera/0/object_mask png')
        object_mask = read_png(res)
        
        # Create empty array for writing
        [m,n] = object_mask.shape[:2]
        res = np.zeros((m,n))
        
        directory = './batch/round' + str(batch) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Convert to greyscale
            
        # Get all object_ids in image
        print("Getting all object_ids in image....")
        id2mask = {}
        for obj_id in scene_objects:
            color = id2color[obj_id]
            mask = match_color(object_mask, [color.R, color.G, color.B], tolerance = 3)
            if mask is not None:
                id2mask[obj_id] = mask
        
        print("Got all object_ids in image.")
        
        # Go through each object id in id2mask
        # Get class of object id from obj_id_to_class
        # check if class is in top 20
        # if it is, then use get the mask from id2mask
        # set those bits to the class number
        
        foundclasses = []
        for obj_id in id2mask.keys():
            thisclass = obj_id_to_class[obj_id]
            if thisclass in top20toint.keys():
                foundclasses.append(thisclass)
                mask = id2mask[obj_id]
                res[mask] = top20toint[thisclass]
        
        print("Found " + str(len(set(foundclasses))) + " classes")
                    
        cv2.imwrite('./batch/round' + str(batch) + '/seg' +                             str(i) + '.png', res*8)
        
        res = client.request('vget /camera/0/lit png')
        normal = read_png(res)
        normal = Image.fromarray(normal)
        
       
        normal.save('./batch/round' + str(batch) + '/pic' +                             str(i) + '.png')
    
        
        print("Images written. ")


# Begin Data Collection (With Normalization)
# ======
# 

for batch in range(1,5):

    # Get random location
    z = 300
    x = random.randint(-5000, 5000)
    y = random.randint(-5000, 5000)

    # Coordinates x, y, z
    client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                    ' ' + str(z)) 

    # Get 10 shots in a series
    angles = []
    a = 0
    while len(angles) < 20:
        angles.append(a)
        a -= 3

    # Increment height sequentially
    heights = []
    height = 300
    while len(heights) < 20:
        heights.append(height)
        height += 50

    for i,angle in enumerate(angles):
        
        print("Batch: " + str(batch) + " , Image: " + str(i))
        
        # x, y, z
        client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                        ' ' + str(heights[i])) 

        # Pitch, yaw, roll
        client.request('vset /camera/0/rotation ' + str(angle) + ' 0 0')

        """
        res = client.request('vget /camera/0/object_mask ./batch/round' + \
                             str(batch) + '/lit' + str(i) + '.png')
        
        print('The image is saved to %s' % res)
        """
        
        # Get Ground Truth
        res = client.request('vget /camera/0/object_mask png')
        object_mask = read_png(res)
        
        # Get all classes in image
        print("Getting all object_ids in image....")
        id2mask = {}
        for obj_id in scene_objects:
            color = id2color[obj_id]
            mask = match_color(object_mask, [color.R, color.G, color.B], tolerance = 3)
            if mask is not None:
                id2mask[obj_id] = mask
        
        print("Got all object_ids in image.")

        """
        Group classes together
        """ 
        class_groups = {}

        # Go through the matched objects
        for idmask in id2mask.keys():

            # Get the class from the neighborhood classes object
            curr_class = obj_id_to_class[idmask]

            # If the class is not in the class_groups map, add it
            if curr_class not in class_groups:
                class_groups[curr_class] = [] 

            # Add the idmask to it's corresponding class
            class_groups[curr_class].append(idmask)


        # Normalize by class
        for cls in class_groups.keys():
            class_base = class_groups[cls][0]
            class_color = id2color[class_base]
            for class_id in class_groups[cls]:
                current_color = id2color[class_id]
                object_mask = swap_color(object_mask, current_color, class_color) 
        
        # Write to file
        print(object_mask.shape)
        normalized_img = Image.fromarray(object_mask)

        directory = './batch/round' + str(batch) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        normalized_img.save('./batch/round' + str(batch) + '/seg' +                             str(i) + '.png')
        
        res = client.request('vget /camera/0/lit png')
        normal = read_png(res)
        print(normal.shape)
        normal = Image.fromarray(normal)
        
        normal.save('./batch/round' + str(batch) + '/pic' +                             str(i) + '.png')
        
        print("Images written. ")





from __future__ import division, absolute_import, print_function
import os, sys, time, re, json
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import StringIO
import json

imread = plt.imread
def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    img = Image.open(StringIO.StringIO(res))
    return np.array(img)

def read_npy(res):
    return np.load(res)

# Matches a color from the object_mask and then returns that region color
def match_color(object_mask, target_color, tolerance=3):
    match_region = np.ones(object_mask.shape[0:2], dtype=bool)
    for c in range(3): # r,g,b
        min_val = target_color[c] - tolerance
        max_val = target_color[c] + tolerance
        channel_region = (object_mask[:,:,c] >= min_val) & (object_mask[:,:,c] <= max_val)
        match_region &= channel_region

    if match_region.sum() != 0:
        return match_region
    else:
        return None
    
# Swap colors method
def swap_color(imgarray, source, dest):
    matched_color = match_color(imgarray, [source.R, source.G, source.B])
    imgarray[:,:,:3][matched_color] = [dest.R, dest.G, dest.B]
    return np.array(imgarray)


# Connect to the game
# ===================
# Load unrealcv python client, do :code:`pip install unrealcv` first.
# 
# 

from unrealcv import client
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
    sys.exit(-1)


# Make sure the connection works well

res = client.request('vget /unrealcv/status')
# The image resolution and port is configured in the config file.
print(res)


# Get objects
# ======================
# Write a json file with the object and their corresponding classes.
# 
# 

scene_objects = client.request('vget /objects').split(' ')
print('Number of objects in this scene:', len(scene_objects))

if '257' in scene_objects:
    scene_objects.remove('257')

obj_id_to_class = {}
for obj_id in scene_objects:
    obj_id_parts = obj_id.split('_')
    class_name = obj_id_parts[0]    
    obj_id_to_class[obj_id] = class_name

# Write JSON file
with open('neighborhood_object_ids.json', 'w') as outfile:
    json.dump(obj_id_to_class, outfile)


# Get object colors
# ======================
# First we create the color object class
# 

class Color(object):
    ''' A utility class to parse color value '''
    regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
    def __init__(self, color_str):
        self.color_str = color_str
        match = self.regexp.match(color_str)
        (self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1,5)]

    def __repr__(self):
        return self.color_str


# Then, we either load from json
# 

id2color = {}
with open('id2color.json') as data_file:
    data = json.load(data_file)

for obj_id in data.keys():
    color_map = data[obj_id]
    color_str = '(R=' + str(color_map['R']) + ',G=' +                 str(color_map['G']) + ',B=' + str(color_map['B']) +                 ',A=' + str(color_map['A']) + ')'
    color = Color(color_str)
    id2color[obj_id] = color


# or We load from the scene
# 

id2color = {} # Map from object id to the labeling color
for i, obj_id in enumerate(scene_objects):
    color = Color(client.request('vget /object/%s/color' % obj_id))
    id2color[obj_id] = color
    print('%d. %s : %s' % (i, obj_id, str(color)))


# Write to JSON if loaded from scene
# 

# Convert to serializable json dictionary
serializable_map = {}
for color_id in id2color.keys():
    curr_color = id2color[color_id]
    color_map = {}
    color_map['R'] = curr_color.R
    color_map['G'] = curr_color.G
    color_map['B'] = curr_color.B
    color_map['A'] = curr_color.A
    serializable_map[color_id] = color_map

# Write to JSON
with open('id2color.json', 'w') as outfile:
    json.dump(serializable_map, outfile)


# Map classes to lists of objects
with open('neighborhood_object_ids.json') as data_file:
    obj_id_to_class = json.load(data_file)
    
classes = {}

for obj_id in obj_id_to_class.keys():
    
    curr_class = obj_id_to_class[obj_id]
    if curr_class not in classes:
        classes[curr_class] = []
    
    classes[curr_class].append(obj_id)

# Write classes to json
with open('neighborhood_classes.json', 'w') as outfile:
    json.dump(classes, outfile) 


# Get top 20 classes from json file
with open('class2count.json') as data_file:
    class2count = json.load(data_file)

from operator import itemgetter
class2countlist = sorted(class2count.items(), key=itemgetter(1), reverse=True)

top20 = []
for cls in class2countlist[0:21]:
    top20.append(cls[0])
    
top20tocolor = {}
for cls in top20:
    top20tocolor[cls] = id2color[classes[cls][0]]

serializable_map = {}
for class_id in top20tocolor.keys():
    curr_color = id2color[classes[class_id][0]]
    color_map = {}
    color_map['R'] = curr_color.R
    color_map['G'] = curr_color.G
    color_map['B'] = curr_color.B
    color_map['A'] = curr_color.A
    serializable_map[class_id] = color_map
    
with open('top20tocolor.json', 'w') as outfile:
    json.dump(serializable_map, outfile)


# Normalize using built in API
counter = 0
for curr_class in classes.keys():
    
    
    object_list = classes[curr_class]
    curr_color = id2color[object_list[0]]
    
    for obj_id in object_list:
        
        if curr_class in top20:
            client.request('vset /object/' + obj_id + '/color ' +                        str(top20.index(curr_class)) + ' 0 0')
        
            print(str(counter) + '. vset /object/' + obj_id + '/color ' +                        str(top20.index(curr_class)) + ' 0 0')
        else:
            client.request('vset /object/' + obj_id + '/color 0 0 0')
            print(str(counter) + '. vset /object/' + obj_id + '/color 0 0 0')
    
        counter += 1


# Begin Data Collection (Without Normalization)
# ======
# 

class2count = {}
for batch in range(1,101):

    # Get random location
    z = 300
    x = random.randint(-5500, 5500)
    y = random.randint(-5500, 5500)

    # Coordinates x, y, z
    client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                    ' ' + str(z)) 

    # Get 10 shots in a series
    angles = []
    a = 0
    while len(angles) < 20:
        angles.append(a)
        a -= 3

    # Increment height sequentially
    heights = []
    height = 300
    while len(heights) < 20:
        heights.append(height)
        height += 50

    for i,angle in enumerate(angles[1:2]):
        
        print("Batch: " + str(batch) + " , Image: " + str(i))
        
        # x, y, z
        client.request('vset /camera/0/location ' + str(x) + ' ' + str(y) +                        ' ' + str(heights[i])) 

        # Pitch, yaw, roll
        client.request('vset /camera/0/rotation ' + str(angle) + ' 0 0')


        # Get Ground Truth
        res = client.request('vget /camera/0/object_mask png')
        object_mask = read_png(res)
        
        # Get all classes in image
        print("Getting all object_ids in image....")
        id2mask = {}
        for obj_id in scene_objects:
            color = id2color[obj_id]
            mask = match_color(object_mask, [color.R, color.G, color.B], tolerance = 3)
            if mask is not None:
                id2mask[obj_id] = mask
        
        # Update class frequency map
        for idmask in id2mask.keys():
            curr_class = obj_id_to_class[idmask]
            if curr_class not in class2count:
                class2count[curr_class] = 1
            else: 
                class2count[curr_class] += 1
               
        # Write to file
        normalized_img = Image.fromarray(object_mask)
        grayscale_img = normalized_img.convert('L')

        directory = './batch/round' + str(batch) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        grayscale_img.save('./batch/round' + str(batch) + '/seg' +                             str(i) + '.png')
        
        
        res = client.request('vget /camera/0/lit png')
        normal = read_png(res)
        normal = Image.fromarray(normal)
        
        normal.save('./batch/round' + str(batch) + '/pic' +                             str(i) + '.png')
        
        print("Images written. ")

with open('class2count.json', 'w') as outfile:
    json.dump(class2count, outfile)


# Load some python libraries
# The dependencies for this tutorials are
# PIL, Numpy, Matplotlib
# 
# 

from __future__ import division, absolute_import, print_function
import os, sys, time, re, json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

imread = plt.imread
def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    img = Image.open(res)
    return np.array(img)

def read_npy(res):
    return np.load(res)


# Connect to the game
# ===================
# Load unrealcv python client, do :code:`pip install unrealcv` first.
# 
# 

from unrealcv import client
client.connect()
if not client.isconnected():
    print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
    sys.exit(-1)


# Make sure the connection works well
# 
# 

res = client.request('vget /unrealcv/status')
# The image resolution and port is configured in the config file.
print(res)


# Ground truth generation
# =======================
# Generate ground truth from this virtual scene
# 
# 

res = client.request('vget /camera/0/object_mask png')
object_mask = read_png(res)
print(object_mask.shape)
res = client.request('vget /camera/0/normal png')
normal = read_png(res)

# Visualize the captured ground truth
"""
plt.imshow(object_mask)
plt.figure()
plt.imshow(normal)
"""


# Get objects
# ======================
# Write a json file with the object and their corresponding classes.
# 
# 

scene_objects = client.request('vget /objects').split(' ')
print('Number of objects in this scene:', len(scene_objects))

data = {}

for obj_id in scene_objects:
    obj_id_parts = obj_id.split('_')
    class_name = obj_id_parts[0]    
    data[obj_id] = class_name

# Write JSON file
import json
with open('neighborhood_classes.json', 'w') as outfile:
    json.dump(data, outfile)


# Get object colors
# ======================
# Map the objects to their labeling colors
# 
# 

# TODO: replace this with a better implementation
class Color(object):
    ''' A utility class to parse color value '''
    regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
    def __init__(self, color_str):
        self.color_str = color_str
        match = self.regexp.match(color_str)
        (self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1,5)]

    def __repr__(self):
        return self.color_str

id2color = {} # Map from object id to the labeling color
for i, obj_id in enumerate(scene_objects):
    color = Color(client.request('vget /object/%s/color' % obj_id))
    id2color[obj_id] = color
    print('%d. %s : %s' % (i, obj_id, str(color)))


# Convert to serializable json dictionary
serializable_map = {}
for color_id in id2color.keys():
    curr_color = id2color[color_id]
    color_map = {}
    color_map['R'] = curr_color.R
    color_map['G'] = curr_color.G
    color_map['B'] = curr_color.B
    color_map['A'] = curr_color.A
    serializable_map[color_id] = color_map

# Write to JSON
with open('id2color.json', 'w') as outfile:
    json.dump(serializable_map, outfile)


# Matches a color from the object_mask and then returns that region color
def match_color(object_mask, target_color, tolerance=3):
    match_region = np.ones(object_mask.shape[0:2], dtype=bool)
    for c in range(3): # r,g,b
        min_val = target_color[c] - tolerance
        max_val = target_color[c] + tolerance
        channel_region = (object_mask[:,:,c] >= min_val) & (object_mask[:,:,c] <= max_val)
        match_region &= channel_region

    if match_region.sum() != 0:
        return match_region
    else:
        return None


# Parse the segmentation mask
# 
# 

id2mask = {}
for obj_id in scene_objects:
    print(obj_id)
    color = id2color[obj_id]
    mask = match_color(object_mask, [color.R, color.G, color.B], tolerance = 3)
    if mask is not None:
        id2mask[obj_id] = mask

# id2mask.keys() are all the matched objects
for idmask in id2mask.keys():
    print(idmask)

# This may take a while
# TODO: Need to find a faster implementation for this


# Group the matched objects
# ===============
# 

class_groups = {}

# Go through the matched objects
for idmask in id2mask.keys():
    
    # Get the class from the data object we had earlier for json file
    curr_class = data[idmask]
    
    # If the class is not in the class_groups map, add it
    if curr_class not in class_groups:
        class_groups[curr_class] = [] 
        
    # Add the idmask to it's corresponding class
    class_groups[curr_class].append(idmask)

for class_group in class_groups.keys():
    print(class_group)


# Create method for swapping colors
# =============
# 

# Swap colors method
def swap_color(imgarray, source, dest):
    matched_color = match_color(imgarray, [source.R, source.G, source.B])
    imgarray[:,:,:3][matched_color] = [dest.R, dest.G, dest.B]
    return np.array(imgarray)


# Test swap_color
other_mask = np.array(object_mask)

class_color = Color('(R=31,G=31,B=31,A=255)')


current_color = Color('(R=31,G=191,B=31,A=255)')
object_mask = swap_color(object_mask, current_color, class_color)


current_color = Color('(R=95,G=63,B=95,A=255)')
object_mask = swap_color(object_mask, current_color, class_color)


current_color = Color('(R=95,G=127,B=95,A=255)')
object_mask = swap_color(object_mask, current_color, class_color)


# Begin normalization by class
# =======
# 

# create copy of object_mask
other_mask = np.array(object_mask)

# Normalize by class
for cls in class_groups.keys():
    
    class_base = class_groups[cls][0]
    class_color = id2color[class_base]
    
    print("class: " + cls + ", color: " + str(class_color))
    
    for class_id in class_groups[cls]:
        
        current_color = id2color[class_id]
        print(current_color)
        object_mask = swap_color(object_mask, current_color, class_color)    


# Show before
before = Image.fromarray(other_mask)
before.show()


# Show after
after = Image.fromarray(object_mask)
after.show()


# Clean up resources
# ==================
# 
# 

client.disconnect()


