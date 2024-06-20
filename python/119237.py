# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# ## Advance lane line finding pipeline
# 1. Camera calibration
# 2. Make undistorted image
# 3. Binarize image through color and gradient threshhold
# 4. Warp undistorted image
# 5. Use sliding window to find left and right lanes
# 6. Use buffer history found lanes to smooth the current lanes
# 7. Caculate the left and right lane curvatures and vehicle offset of the center of the lane
# 8. Unwarp processed image back and make movie
# 
# ---
# ## Define CameraCalibrator class
# 

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# store camera calibration parameters in ./camera_cal/calibrated_data.p
CALIBRATED_DATA_FILE = './camera_cal/calibrated_data.p'

class CameraCalibrator:
    def __init__(self, glob_regex, x_corners, y_corners, init_coef=True):
 
        
        # images used for camera calibration
        self.calibration_images = glob_regex
        
        # The number of horizontal corners in calibration images
        self.x_corners = x_corners
        
        # The number of vertical corners in calibration images
        self.y_corners = y_corners
        self.object_points = []
        self.image_points = []
        self.chessboards = []
        self.calibrated_data = {}
        if not init_coef:
            self.calibrate_via_chessboards()

        self.coef_loaded = False
  

    def calibrate_via_chessboards(self):

        object_point = np.zeros((self.x_corners * self.y_corners, 3), np.float32)
        object_point[:,:2] = np.mgrid[0:self.x_corners, 0:self.y_corners].T.reshape(-1, 2)

        for idx, file_name in enumerate(self.calibration_images):
            image = mpimg.imread(file_name)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image,(self.x_corners, self.y_corners),None)
            if ret:
                self.object_points.append(object_point)
                self.image_points.append(corners)
                
                # Draw and display the corners
                chessboard_image = np.copy(gray_image)
                cv2.drawChessboardCorners(chessboard_image, (9,6), corners, ret)
                self.chessboards.append(chessboard_image)

        h, w = image.shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.image_points, (w, h), None, None)

        self.calibrated_data = {'mtx': mtx, 'dist': dist}

        with open(CALIBRATED_DATA_FILE, 'wb') as f:
            pickle.dump(self.calibrated_data, file=f)

        self.coef_loaded = True

    def undistort(self, image):

        if not os.path.exists(CALIBRATED_DATA_FILE):
            raise Exception('Camera calibration data file does not exist at ' + CALIBRATED_DATA_FILE)

        if not self.coef_loaded:

            with open(CALIBRATED_DATA_FILE, 'rb') as fname:
                self.calibrated_data = pickle.load(file=fname)

            self.coef_loaded = True

        return cv2.undistort(image, self.calibrated_data['mtx'], self.calibrated_data['dist'],None, self.calibrated_data['mtx'])


# ##  Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# 1. Class CameraCalibrator has construntor,calibrate_via_chessboards,and undistort method. 
# 2. I use cv2.findChessboardCorners to find inner coners in chessboard and use cv2.calibrateCamera to get calibration matix and distortion coefficients.
# 

# Use glob to get a list of calibration images
calibration_images = glob.glob('./camera_cal/calibration*.jpg')

# Here init_coef  set false to call calibrate_via_chessboards() at first time 
# and store  calibration matix and distortion coefficients in './camera_cal/calibrated_data.p'
calibrator = CameraCalibrator(calibration_images, 9, 6, init_coef = False)


get_ipython().magic('matplotlib inline')
# Display the chessboard with 9X6 corners drawn
plt.figure(figsize=(10,5))
plt.imshow(calibrator.chessboards[3])
plt.show()


# ## Apply a distortion correction to raw images.
# 

# Take example distorted image and undistort it using saved camera coefficients
distorted_image = './test_images/test4.jpg'
distorted_image = cv2.imread(distorted_image)
undistorted_image = calibrator.undistort(distorted_image)

# Display both distorted and undistorted images
plt.figure(figsize=(12,7))
plt.subplot(1, 2, 1)
plt.title('Distorted test4.jpg')
plt.imshow(distorted_image)

plt.subplot(1, 2, 2)
plt.imshow(undistorted_image)
plt.title('Undistorted test4.jpg')

plt.show()


UNDIST_IMAGES_LOCATION = './output_images/test_images_undistorted/'
TEST_IMAGES_LOCATION = './test_images/'
images_loc = os.listdir(TEST_IMAGES_LOCATION)

for image_loc in images_loc:
    corred_image_file = UNDIST_IMAGES_LOCATION + image_loc
    distorted_image_location = TEST_IMAGES_LOCATION + image_loc
    distorted_image_location = cv2.imread(distorted_image_location)
    corrected_image = calibrator.undistort(distorted_image_location)
    cv2.imwrite(corred_image_file, corrected_image)


# ## Apply a perspective transform to warp the image ("birds-eye view").
# 
# On test_images/straight_lines2.jpg, I choose the 4 points [277, 670], [582, 457], [703, 457], [1046, 670] to get perspectivet_ransformer; following lines are 4 step pipeline to do it .
# 1. define 4 source points src = np.float32([[277, 670], [582, 457], [703, 457], [1046, 670]])
# 2. define 4 destination points dst = np.float32([[277, 670], [277,0], [1046,0], [1046,670]])
# 3. use cv2.getPerspectiveTransform() to get M, the transform matrix
# 4. use cv2.warpPerspective() to warp your image to a top-down view
# 

get_ipython().magic('matplotlib inline')
    
def warp(img):

    img_size = (img.shape[1],img.shape[0])
    
    corners = np.float32([[277, 670], [582, 457], [703, 457], [1046, 670]])

    src = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst = np.float32([[277, 670], [277,0], [1046,0], [1046,670]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv

# 
fname = 'test_images/straight_lines2.jpg'
image = mpimg.imread(fname)
undistorted_image = calibrator.undistort(image)
cv2.imwrite('test_images/straight_lines2_undistorted.png',undistorted_image, [int( cv2.IMWRITE_JPEG_QUALITY), 95])

# Choose  
pts = np.array([[277, 670], [582, 457], [703, 457], [1046, 670]], np.int32)
pts = pts.reshape((-1,1,2))
imd = np.copy(undistorted_image)
imd = cv2.polylines(imd,[pts],True,(255,0,0),2)

binary_warped, perspective_M, Minv  = warp(imd)

plt.figure(figsize=(12, 7))

plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Undistorted straight_lines2.jpg' ,fontsize=20)
plt.imshow(imd)

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Binarized straight_lines2.jpg', fontsize=20)
plt.imshow(binary_warped,cmap='gray')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()


# ## Use color transforms, gradients, etc., to create a thresholded binary image.
# We use the following two threshhold binary filters:
# 1. Apply Sobel operator in X direction
# 2. Convert image to HLS color space, and use the L and S channel threshhold to filter cause HLS scheme is more reliable when it comes to find out lane lines
# 

get_ipython().magic('matplotlib inline')

# This method is used to reduce the noise of binary images.
def noise_reduction(image, threshold=4):
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(image, ddepth=-1, kernel=k)
    image[nb_neighbours < threshold] = 0
    return image

def binary_threshold_filter(channel, thresh = (200, 255), on = 1):
    binary = np.zeros_like(channel)
    binary[(channel > thresh[0]) & (channel <= thresh[1])] = on
    return binary
    
def binary_pipeline(image, 
					hls_s_thresh = (170,255),
					hls_l_thresh = (30,255),
					hls_h_thresh = (15,100),
					sobel_thresh=(20,255),
					mag_thresh=(70,100),
					dir_thresh=(0.8,0.9),
					r_thresh=(150,255),
					u_thresh=(140,180),
					sobel_kernel=3):

    # Make a copy of the source iamge
    image_copy = np.copy(image)

    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

    r_thresh = (150,255)
	# RGB colour
    R = image_copy[:,:,0]
    G = image_copy[:,:,1]
    B = image_copy[:,:,2]
    rbinary = binary_threshold_filter(R, r_thresh)
    
    u_thresh = (140,180)
    # YUV colour
    yuv = cv2.cvtColor(image_copy, cv2.COLOR_RGB2YUV)
    Y = yuv[:,:,0]
    U = yuv[:,:,1]
    V = yuv[:,:,2]
    ubinary = binary_threshold_filter(U, u_thresh)

    # Convert RGB image to HLS color space.
    # HLS more reliable when it comes to find out lane lines
    hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]
    h_channel = hls[:, :, 0]
    # We generated a binary image using S component of our HLS color scheme and provided S,L,H threshold
    s_binary = binary_threshold_filter(s_channel,hls_s_thresh)
    l_binary = binary_threshold_filter(l_channel,hls_l_thresh)
    h_binary = binary_threshold_filter(h_channel,hls_h_thresh)

    # We apply Sobel operator in X,Y direction and calculate scaled derivatives.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = binary_threshold_filter(scaled_sobel,sobel_thresh)

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)
    scaled_sobel = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    sybinary = binary_threshold_filter(scaled_sobel,sobel_thresh)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    mag_binary = binary_threshold_filter(gradmag, mag_thresh)

    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = binary_threshold_filter(absgraddir, dir_thresh)

    # Return the combined binary image
    binary = np.zeros_like(sxbinary)
    binary[(((l_binary == 1) & (s_binary == 1) | (sxbinary == 1)) ) ] = 1
    binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')

    return noise_reduction(binary)
                           
# Use distortion image
fname = 'test_images/test4.jpg'
img_shadow = mpimg.imread(fname)
undistorted_shadow = calibrator.undistort(img_shadow)
combined_binary_shadow = binary_pipeline(undistorted_shadow)

plt.figure(figsize=(12, 7))

plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Undistorted test4.jpg' ,fontsize=20)
plt.imshow(undistorted_shadow)

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Binarized test4.jpg', fontsize=20)
plt.imshow(combined_binary_shadow)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()


# ## Warp undistorted birarized image
# 

binary_images = []
for i in range(1,7):
    
    fname = 'test_images/test'+str(i)+'.jpg'
    img = mpimg.imread(fname)
    undistorted = calibrator.undistort(img)
    undistorted_unwarped, _, _ = warp(undistorted)
    combined_binary = binary_pipeline(undistorted_unwarped)
    img_size = (undistorted.shape[1],undistorted.shape[0])

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    f.tight_layout()
    ax1.imshow(undistorted_unwarped)
    ax1.set_title('Undistorted , warped ' + fname, fontsize=15)
    ax2.imshow(combined_binary)
    ax2.set_title('Undistorted , warped and binarized '+ fname, fontsize=15)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    binary_images.append(combined_binary)


OUTPUT_DIR = './output_images/test_images_binary/'
INPUT_DIR = './output_images/test_images_undistorted/'

for file in os.listdir(INPUT_DIR):
    saved_undistorted_img = mpimg.imread(INPUT_DIR + file)
    binary_img = binary_pipeline(saved_undistorted_img)
    cv2.imwrite(OUTPUT_DIR + file, binary_img)


# ## Use sliding window to find left and right lanes
# 

# Choose test6.jpg to do experiment
warped_image = binary_images[5] 
print(warped_image.shape)
plt.imshow(warped_image)
plt.axis('off')
plt.show()


# ### Use histogram of pixels to find the bottm point of left and right lane lines of image
# 

#Bottom half region of image  of 0 channel
histogram = np.sum(warped_image[warped_image.shape[0] // 2:, :, 0], axis=0)
plt.figure(figsize=(8, 3))
plt.plot(histogram)
plt.show()


# get midpoint of the histogram  == half of width
midpoint = np.int(histogram.shape[0] / 2)

# get left and right half points of the histogram
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

print('Peak point of left half: {}'.format(leftx_base))
print('Peak point of right half: {}'.format(rightx_base))


# ### Detect lane line pixels and fit to lane curve
# 

# Choose the number of sliding windows
nwindows = 9

# Set height of sliding window
window_height = np.int(warped_image.shape[0] / nwindows)

# Identify the x and y positions of all nonzero pixels in the image
nonzero = warped_image.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base

# Set the width of the windows +/- margin
margin = 100

# Set minimum number of pixels found to recenter window
min_num_pixels = 50

# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

for window in range(nwindows):
    
    # Identify window boundaries in x and y (and right and left)
    win_y_low = warped_image.shape[0] - (window + 1) * window_height
    win_y_high = warped_image.shape[0] - window * window_height

    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    # Search pixels in sliding windows
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    if len(good_left_inds) > min_num_pixels:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > min_num_pixels:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the ndarrays of indices
left_lane_array = np.concatenate(left_lane_inds)
right_lane_array = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_array]
lefty = nonzeroy[left_lane_array]
rightx = nonzerox[right_lane_array]
righty = nonzeroy[right_lane_array]

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values for plotting
ploty = np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0])
fit_leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
fit_rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


warped_image[nonzeroy[left_lane_array], nonzerox[left_lane_array]] = [255, 0, 0]
warped_image[nonzeroy[right_lane_array], nonzerox[right_lane_array]] = [0, 0, 255]
plt.imshow(warped_image)
plt.plot(fit_leftx, ploty, color='yellow')
plt.plot(fit_rightx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()


# ### Polyfill the found left and right lanes and unwarp onto original distorted image 
# 

fname = 'test_images/test4.jpg'
img = mpimg.imread(fname)
image_size = img.shape
undistorted = calibrator.undistort(img)
undistorted_unwarped, _, _ = warp(undistorted)
binary_warped = binary_pipeline(undistorted_unwarped)
    
# Create an image to draw the lines on
warp_zero = np.zeros_like(undistorted)
fit_y = np.linspace(0, warp_zero.shape[0] - 1, warp_zero.shape[0])


# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([fit_leftx, fit_y]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fit_y])))])

pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(warp_zero, Minv, (undistorted.shape[1], undistorted.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
plt.imshow(result)


# ## Measuring Curvature
# 

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
# For each y position generate random x position within +/-50 pix
# of the line base position in each case (x=200 for left, and x=900 for right)
leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                              for y in ploty])
rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                for y in ploty])

leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


# Fit a second order polynomial to pixel positions in each fake lane line
left_fit = np.polyfit(ploty, leftx, 2)
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fit = np.polyfit(ploty, rightx, 2)
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Plot up the fake data
mark_size = 3
plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, ploty, color='green', linewidth=3)
plt.plot(right_fitx, ploty, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images


# ##  calculate the radius of curvature in pixels
# 

# Define y-value where we want radius of curvature
# I'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
print('Left lane curvature( in pixel) : ',left_curverad)
print('Right lane curvature( in pixel) : ',right_curverad)


# ### Calculation of radius of curvature after correcting for scale in x and y
# 

# f(y)= A*power(y,2)+B*y+C
# y= image_size[0]  ==> f(y) = intersection points at the bottom of our imag
left_intercept = left_fit[0] * img_size[0] ** 2 + left_fit[1] * img_size[0] + left_fit[2]
right_intercept = right_fit[0] * img_size[0] ** 2 + right_fit[1] * img_size[0] + right_fit[2]

# Caculate the distance in pixels between left and right intersection points
road_width_in_pixels = right_intercept - left_intercept
print('pixels between left and right intersection points ',road_width_in_pixels)

# Since average highway lane line width in US is about 3.7m
# Assume lane is about 30 meters long and 3.7 meters wide
# we calculate length per pixel and ensure "order of magnitude" correct
xm_per_pix = 3.7 / road_width_in_pixels
ym_per_pix = 30 / image_size[0]

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
# Calculate the new radii of curvature
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

# Now our radius of curvature is in meters
print('Left lane curvature( in metes) : ',left_curverad,' m')
print('Right lane curvature( in meters) : ',right_curverad,' m')

# lane deviation
calculated_center = (left_intercept + right_intercept) / 2.0
lane_deviation = (calculated_center - img_size[1] / 2.0) * xm_per_pix
print('lain deviation ',lane_deviation,'m')


# ### Sanity Check
# 

# Here means something wrong with lane finding 
assert road_width_in_pixels > 0, 'Road width in pixel must be positive!!'


# ## Use buffer history found lanes to smooth the current lanes
# I use a buffer in Line class to store the previous found 10 lane lines and use the median value of buffers as the unwarp ones. See line 431 ~ line 438 in code file 'Advanced_Lane_Finding.py'
# 

# ## Unwarp processed image back and make movie
# 

# I create **class Line** in file **Advanced_Lane_Finding.py** to include all the methods we discussed above;
# The key method **image_process_pipeline** take parameter image to which implement the following pipeline:
# 1. Make undistorted image
# 2. Warp undistorted image
# 3. Binarize image through color and gradient threshhold
# 4. Use sliding window to find left and right lanes
# 5. Use buffer history found lanes to smooth the current lanes
# 6. Caculate the left and right lane curvatures and vehicle offset of the center of the lane
# 7. Unwarp processed image back and display
# 

from moviepy.editor import VideoFileClip
from Advanced_Lane_Finding import *

line = Line()
input_file = './project_video.mp4'
output_file = './processed_project_video.mp4'
clip = VideoFileClip(input_file)
out_clip = clip.fl_image(line.image_process_pipeline)
out_clip.write_videofile(output_file, audio=False)


from IPython.display import HTML
output_file = './processed_project_video.mp4'
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output_file))








# # Vehicle Detection and Tracking
# 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from skimage.feature import hog
get_ipython().magic('matplotlib inline')


# ## Histogram of Oriented Gradients (HOG)
# 
# Fist I define functions for features extraction (HOG, binned color and color histogram features). 
# 

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(16, 16)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# 1. The **extract_features** is main feature extraction function which augment dataset by horizontal image flipping.
# 2. The **image_features_fun** is helper function.  
# 

def image_features_fun(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel,color_space='LUV'):
    file_features = []
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
        else:
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_HSV2RGB)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_LUV2RGB)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_HLS2RGB)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_YUV2RGB)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(feature_image, cv2.COLOR_YCrCb2RGB)
                
            feature_image = cv2.cvtColor(feature_image, cv2.COLOR_RGB2GRAY)
            hog_features = get_hog_features(feature_image[:,:], orient, 
                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        
        # Append the new feature vector to the features list
        file_features.append(hog_features)
    return file_features

# Define a function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each imageone by one
        image = cv2.imread(file) 
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else: feature_image = np.copy(image)      
        file_features = image_features_fun(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel,color_space)
        features.append(np.concatenate(file_features))
        # Augment train data with flipped images
        feature_image=cv2.flip(feature_image,1) 
        file_features = image_features_fun(feature_image, spatial_feat, hist_feat, hog_feat, hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel,color_space)
        features.append(np.concatenate(file_features))
    return features 


# ### Load Train dataset
# I use train dataset [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [not-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) provided by Udacity.
# 

# Read in cars and notcars
images = glob.glob('train_data/*/*/*')
cars = []
notcars = []
for image in images:
    if 'non' in image:
        notcars.append(image)
    else:
        cars.append(image)
# Balance number of images of both vehicle and non-vehicle class.
# Reduce CPU time.
#sample_size = 4000
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]
print('Car samples have images：',len(cars))
print('Non-car sampples have images：',len(notcars))


# Here is an example of one of each of the vehicle and non-vehicle images:
# 

plt.figure(figsize=(6,3))
plt.subplot(1, 2, 1)
plt.title('Cars')
plt.imshow(mpimg.imread(cars[1000]))

plt.subplot(1, 2, 2)
plt.imshow(mpimg.imread(notcars[1000]))
plt.title('Non cars')
plt.show()


# ### HOG feature visualization
# Here is an example for HOG parameters of orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2):
# 

car_image=mpimg.imread(cars[1200])
gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
features, hog_image = get_hog_features(gray, 16, 8, 2,vis=True, feature_vec=True)

notcar_image=mpimg.imread(notcars[1200])
notcar_gray = cv2.cvtColor(notcar_image, cv2.COLOR_BGR2GRAY)
features, not_hog_image = get_hog_features(notcar_gray, 16, 8, 2,vis=True, feature_vec=True)

fig = plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(car_image, cmap='gray')
plt.title('Car Image')

plt.subplot(2,2,2)
plt.imshow(hog_image, cmap='gray')
plt.title('Car HOG Feature')
plt.show

plt.subplot(2,2,3)
plt.imshow(notcar_image, cmap='gray')
plt.title('Non-Car Image')

plt.subplot(2,2,4)
plt.imshow(not_hog_image, cmap='gray')
plt.title('Non-Car HOG Feature')
plt.show


# ## Explain how you settled on your final choice of HOG parameters
# 

# 1. The most important for detection is speed which depends on the speed of feature extraction speed. 
# So I tried several HOG parameters combination. Here we can see color space 'LUV' and hog channel 0 make fastest   extraction.
# 2. But after several classifier tests, we found the LUV color space and hog channel 0 made better prediction accuracy!!
# 

# Define parameters for feature extraction
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


for colorspace in ('RGB','HSV','LUV','HLS','YUV','YCrCb'):
 
    for ch in (0,1,2):
        print('color space: ',colorspace,' HOG channel:',ch)
        t=time.time()   
        car_features = extract_features(cars[0:100], color_space=colorspace, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=ch, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

        notcar_features = extract_features(notcars[0:100], color_space=colorspace, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=ch, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        t2 = time.time()
        
        print(round(t2-t, 2), 'Seconds to extract featrues')


# ###  Train classifier
# 
# The following code creates features for both cars and non-cars train datasets. I nomalized data features by the method **sklearn.StandardScaler()**. The data is splitted into thaining and testing subsets using **train_test_split**(80% and 20%).Then I train the classifier using linear svc and with HOG parameters : 8 orientations 8 pixels per cell and 2 cells per block.
# 

# Define parameters for feature extraction
color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)     

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X) 
# Apply the scaler to X
scaled_X = X_scaler.transform(X) 

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)))) 

# Split up data into randomized training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=22)

print('orientations:',orient)
print('pixels per cell:', pix_per_cell)
print('cells per block',cell_per_block)

print('Feature vector length:', len(X_train[0]))

# Use a linear SVC
svc = LinearSVC()  
# Check the training time for the SVC
t=time.time() 
# Train the classifier
svc.fit(X_train, y_train) 
t2 = time.time()
print('Trainning time is:',round(t2-t, 2),' seconds')
# Test dataset model prediction accuracy
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 5)) 


# ## Slide window
# 
# Here we define a sliding window function **slide_window** to generate a list of boxes and a **draw_boxes** to draw the  boxes on an image. The following functions are from the Udacity's lectures because they just work and have good performance.
# 

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(128, 128), xy_overlap=(0.85, 0.85)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes on an image
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img) # Make a copy of the image
    for bbox in bboxes: # Iterate through the bounding boxes
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def single_img_features(img, color_space='RGB', spatial_size=(16, 16),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)
    #9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='LUV', 
                    spatial_size=(16, 16), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


# ### Classifier test
# 
# 1. Here we test the calssifier on the test images. I decided to use 128X128 sliding window size and overlap rate 0.85  to implement **slide_window** function after several experiments .
# 2. We can see that the speed is a little slow.
# 

# A function to show an image
def show_img(img):
    #if read image using cv2.imread
    if len(img.shape)==3: 
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else: # Grayscale image
        plt.figure()
        plt.imshow(img, cmap='gray')
        

for image_p in glob.glob('test_images/test*.jpg'):
    t=time.time() # Start time
    image = cv2.imread(image_p)
    draw_image = np.copy(image)
    # Here we define a specific range area for test
    windows = slide_window(image, x_start_stop=[880, None], y_start_stop=[400, None], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    windows += slide_window(image, x_start_stop=[0, 400], y_start_stop=[400, None], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    windows += slide_window(image, x_start_stop=[400, 1280], y_start_stop=[400, 550], 
                    xy_window=(64, 64), xy_overlap=(0.75, 0.75))

    hot_windows = []
    hot_windows += (search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat))
    print('Time of process test images: ',round(time.time()-t, 2),' seconds.')
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    show_img(window_img)
    


# ## Improved Sliding Window
# To increase the performance of detection,we must decide the smallest windows and searching areas.So we only scan some parts of image where the cars will show up. 
# ### ROI settings
# We find new cars which in the red boxes area and far cars which in green boxes area.
# 

image = cv2.imread('test_images/test6.jpg')
windows = slide_window(image, x_start_stop=[800, 1280], y_start_stop=[400, 650], 
                    xy_window=(128, 128), xy_overlap=(0.85, 0.85))
windows += slide_window(image, x_start_stop=[0, 600], y_start_stop=[400, 650], 
                    xy_window=(128, 128), xy_overlap=(0.85, 0.85))
window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6) 
windows = slide_window(image, x_start_stop=[400, 1000], y_start_stop=[400, 500], 
                    xy_window=(64, 64), xy_overlap=(0.75, 0.75))
window_img = draw_boxes(window_img, windows, color=(0, 255, 0), thick=6)                    
show_img(window_img)


# #### Advanced Lane line function
# 
# Here we load lane finding function from the [Advanced Lane lines](https://github.com/wuqianliang/ND013/tree/master/advanced%20lane%20finding).
# 

from Advanced_Lane_Finding  import *
line=Line()
show_img(line.image_process_pipeline(cv2.imread('test_images/test6.jpg')))


# ## Multiple Detections & False Positives
# 

# ### Hog Sub-sampling Window Search
# In order to detect several deferent size and position of cars in images and **speed up** the detection process, we define a function **find_cars()** to generate windows with cars in it by a specific scale such that we could run this same function multiple times for different scale values to generate multiple-scaled search windows.
# 

def convert_color(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def find_cars(img, ystart, ystop, xstart, xstop, scale, step, 
              svc, X_scaler, orient=8, pix_per_cell=8, cell_per_block=2, spatial_size=(16, 16), hist_bins=32,_window=64):
    boxes = []
    draw_img = np.zeros_like(img)   
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale))) 
    
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    #print('nfeat_per_block:',nfeat_per_block)
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = _window
    nblocks_per_window = (window // pix_per_cell) -1
    #print('nblocks_per_window:',nblocks_per_window)
    # Instead of overlap, define how many cells to step
    cells_per_step = step  
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            # Extract the image patch
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))        
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)+xstart
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((int(xbox_left), int(ytop_draw+ystart)),(int(xbox_left+win_draw),int(ytop_draw+win_draw+ystart))))
    return boxes


# ### Remove False Positives
# 1. Here we define heatmap filtering functions suggested by lectures to remove false positive boxes. We filter all found windows by a heatmap threshhold.
# 2. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.
# 

from scipy.ndimage.measurements import label

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


# Tests of filter for false positives and some method for combining overlapping bounding boxes.
# 

for file in glob.glob('test_images/test*.jpg'):
    t=time.time() # Start time
    image = cv2.imread(file)
    
    img = np.copy(image)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Here we define a specific range area for test
    windows = slide_window(img, x_start_stop=[800, None], y_start_stop=[400, None], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    windows += slide_window(img, x_start_stop=[0, 400], y_start_stop=[400, None], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    windows += slide_window(img, x_start_stop=[400, 1280], y_start_stop=[400, 550], 
                    xy_window=(64, 64), xy_overlap=(0.75, 0.75))

    hot_windows = []
    hot_windows += (search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat))
    #print('Time of process test images: ',round(time.time()-t, 2),' seconds.')
    window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)
    
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    fig = plt.figure(figsize=(18,10))
    plt.subplot(1,3,1)
    plt.imshow(window_img, cmap='gray')
    plt.title('Drawing boxes')
    plt.subplot(1,3,2)
    plt.imshow(heatmap, cmap='gray')
    plt.title('heatmap')
    plt.subplot(1,3,3)
    plt.imshow(labels[0], cmap='gray')
    plt.title('lables')
    plt.show()


# ## Detection and tracking process pipeline
# 

def image_process_pipeline(image):
    
    img = np.copy(image)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    boxes = []
        
    '''
    # Here is the working sliding window tech, but very slow when process video 
    if True:
        # Use sliding windows in ROI area
        windows = slide_window(image, x_start_stop=[880, None], y_start_stop=[400, None], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
        windows += slide_window(image, x_start_stop=[0, 400], y_start_stop=[400, None], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
        
        windows += slide_window(img, x_start_stop=[400, 1280], y_start_stop=[350, 650], 
                    xy_window=(64, 64), xy_overlap=(0.75, 0.75))
        boxes += (search_windows(img, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat))
    '''
    if True:
        # Right side 
        boxes = find_cars(img, 400, 650, 800, 1280, 2.5, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        boxes += find_cars(img, 400, 500, 800, 1280, 1, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        # Left side
        boxes += find_cars(img, 400, 650, 0, 400, 2.5, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        boxes += find_cars(img, 400, 500, 0, 400, 1, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        # For far cars
        boxes += find_cars(img, 350, 400, 400, 800, 0.75, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
     
    # Add heat to each box in box list
    heat = add_heat(heat,boxes)
    
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    # Debug boxes
    #draw_img = draw_boxes(draw_img, boxes, color=(0, 255, 0), thick=4)
    
    return cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)


# #### Example of video process pipeline:
# 1. Undistort image and draw lane line
# 2. Detect cars and get process time which is more fast than the previous sliding window tech
# 

for ima in glob.glob('test_images/test*.jpg'):
    image1 = cv2.imread(ima)
    
    image2 = line.image_process_pipeline(image1)
    image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    t=time.time() # Start time
    show_img(image_process_pipeline(image3))
    print('Time of process test images: ',round(time.time()-t, 2),' seconds.')


# ## Video processing
# 
# Here we process the test_video.mp4 and project_video.mp4 using VideoFileClip.
# 

from moviepy.editor import VideoFileClip
from object_detect_yolo import YoloDetector
def process_image(image):
 
    t=time.time() # Start time
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image2 = line.image_process_pipeline(image)
    image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(image_process_pipeline(image3), cv2.COLOR_BGR2RGB)
    print('Time of process test images: ',round(time.time()-t, 2),' seconds.')

output_v = 'processed_test_video.mp4'
clip1 = VideoFileClip("test_video.mp4")
clip = clip1.fl_image(process_image)
get_ipython().magic('time clip.write_videofile(output_v, audio=False)')


output_v = 'processed_project_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip = clip1.fl_image(process_image)
get_ipython().magic('time clip.write_videofile(output_v, audio=False)')


