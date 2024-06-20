import cv2
import numpy as np
import matplotlib.pyplot as plt


# # Edge Detection
# One of the fundamental operations in image processing. It helps use reduce amount of data to process while maintaining the structural aspect of the image.
# 1. Sobel Edge Detection (First order detection)
# 2. Roberts Edge Detection (First order detection)
# 3. Prewitt Edge Detection (First order detection)
# 4. Kirsch Edge Detection (First order detection)
# 5. Nevatia/Babu Edge Detection (First order detection)
# 6. Laplacian Edge Detection (Second order detection)
# 7. Canny Edge Detection
# 
# 
# - First Order Detection - Very sensitive to noise and produce thicker edges (Maximum Detection).
# - Second Order Detection - Less sensitive to noise. (Zero-Crossing Detection)
# 
# $$x(t) \longleftrightarrow X(j\omega)$$
# $$\frac{d[x(t)]}{dx} \longleftrightarrow j\omega X(j\omega)$$
# 
# As we can see, differentiation amplifies noise, smoothening is suggested prior to applying edge detection.
# 
# Choosing the optimal edge detection depends on the edge profile of the object to be detected.
# 

# ## Sobel Edge Detection
# A gradient based method, calculating first order derivatives of the image separately for the X and Y axes. It uses two 3x3 kernels which are convolved with the original image to calculate approximations of the derivatives. It is more sensitive to diagonal edges than to the horizontal and vertical edges.
# $$\text{X kernel} = \begin{bmatrix}-1 & 0 & +1\\-2 & 0 & +2\\-1 & 0 & +1\end{bmatrix}$$
# 
# $$\text{Y kernel} = \begin{bmatrix}+1 & +2 & +1\\0 & 0 & 0\\-1 & -2 & -1\end{bmatrix}$$
# Edge shown by jump in intensity (1-D image)<img src="../resources/sobel1.jpg">Edge more easy to observe after taking first derivative (maximum)<img src="../resources/sobel2.jpg">
# Function: <b>cv2.Sobel(image, depth, dx, dy[, dst[, ksize...]])</b>
# 

image = cv2.imread('../resources/messi.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (3, 3), 0)

sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

# sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')

fig.add_subplot(2, 2, 2)
plt.imshow(sobelx, cmap='gray')

fig.add_subplot(2, 2, 3)
plt.imshow(sobely, cmap='gray')

fig.add_subplot(2, 2, 4)
plt.imshow(sobel, cmap='gray')

plt.show()


# ## Roberts Edge Detection
# It uses two 2x2 kernels. One kernel is simply the other rotated by 90 degrees.
# $$\text{X kernel} = \begin{bmatrix}+1 & 0\\0 & -1\end{bmatrix}$$
# 
# $$\text{Y kernel} = \begin{bmatrix}0 & +1\\-1 & 0\end{bmatrix}$$
# 
# Responds maximally to edges running at 45° to the pixel grid. It is very quick to compute. Since it uses such a small kernel, it's highly sensitive to noise. It also produces weak responses to genuine edges.
# 

from scipy import ndimage

roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

img = np.asarray(blurred, dtype="int32")

vertical = ndimage.convolve(img, roberts_y)
horizontal = ndimage.convolve(img, roberts_x)

robert = np.sqrt(np.square(vertical) + np.square(horizontal))

plt.imshow(robert, cmap='gray')
plt.show()


# ## Prewitt Edge Detection
# Uses two 3x3 kernels to calculate approximations of the derivatives.
# $$\text{X kernel} = \begin{bmatrix}+1 & 0 & -1\\+1 & 0 & -1\\+1 & 0 & -1\end{bmatrix}$$
# 
# $$\text{Y kernel} = \begin{bmatrix}+1 & +1 & +1\\0 & 0 & 0\\-1 & -1 & -1\end{bmatrix}$$
# 
# As compared to Sobel, the Prewitt masks are simpler to implement but are very sensitive to noise. It is more sensitive to horizontal and vertical edges
# 

prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

horizontal = ndimage.convolve(img, prewitt_x)
vertical = ndimage.convolve(img, prewitt_y)

prewitt = np.sqrt(np.square(horizontal) + np.square(vertical))

plt.imshow(prewitt, cmap='gray')
plt.show()


# ## Kirsch Edge Detection
# Uses a single 3x3 kernel mast and rotates it in 45 degrees increments through all the 8 compass directions.
# $$kernel = \begin{bmatrix}+5 & +5 & +5\\-3 & 0 & -3\\-3 & -3 & -3\end{bmatrix}$$
# <img src="../resources/compass.jpg">
# It's good detector but requires a lot of computations.
# 
# Nevatia/Babu Edge Detection is similar to Kirsch, the matirx rotates in 30 degree increments from 0->30->60->90->120->150.
# 

def rotate_45(array):
    result = np.zeros_like(array)
    for i in range(3):
        for j in range(3):
            if i == 0 and j == 0:   result[i+1][j] = array[i][j]
            elif i == 2 and j == 0: result[i][j+1] = array[i][j]
            elif i == 2 and j == 2: result[i-1][j] = array[i][j]
            elif i == 0 and j == 2: result[i][j-1] = array[i][j]
            elif i == 0:            result[i][j-1] = array[i][j]
            elif j == 0:            result[i+1][j] = array[i][j]
            elif i == 2:            result[i][j+1] = array[i][j]
            elif j == 2:            result[i-1][j] = array[i][j]
    return result

kernel_n = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
kernel_nw = rotate_45(kernel_n)
kernel_w = rotate_45(kernel_nw)
kernel_sw = rotate_45(kernel_w)
kernel_s = rotate_45(kernel_sw)
kernel_se = rotate_45(kernel_s)
kernel_e = rotate_45(kernel_se)
kernel_ne = rotate_45(kernel_e)

d1 = ndimage.convolve(img, kernel_n)
d2 = ndimage.convolve(img, kernel_nw)
d3 = ndimage.convolve(img, kernel_w)
d4 = ndimage.convolve(img, kernel_sw)
d5 = ndimage.convolve(img, kernel_s)
d6 = ndimage.convolve(img, kernel_se)
d7 = ndimage.convolve(img, kernel_e)
d8 = ndimage.convolve(img, kernel_ne)

kirsch = np.sqrt(np.square(d1) + np.square(d2) + np.square(d3) + np.square(d4) +  
                 np.square(d5) + np.square(d6) + np.square(d7) + np.square(d8))

fig = plt.figure()
fig.set_size_inches(36, 20)

fig.add_subplot(3, 4, 1)
plt.imshow(d1, cmap='gray')

fig.add_subplot(3, 4, 2)
plt.imshow(d2, cmap='gray')

fig.add_subplot(3, 4, 3)
plt.imshow(d3, cmap='gray')

fig.add_subplot(3, 4, 4)
plt.imshow(d4, cmap='gray')

fig.add_subplot(3, 4, 5)
plt.imshow(d5, cmap='gray')

fig.add_subplot(3, 4, 6)
plt.imshow(d6, cmap='gray')

fig.add_subplot(3, 4, 7)
plt.imshow(d7, cmap='gray')

fig.add_subplot(3, 4, 8)
plt.imshow(d8, cmap='gray')

fig.add_subplot(3, 4, 10)
plt.imshow(img, cmap='gray')

fig.add_subplot(3, 4, 11)
plt.imshow(kirsch, cmap='gray')

plt.show()


# ## Laplacian Edge Detection
# Uses one 3x3 kernel and calculates second order derivatives in a single pass.
# $$kernel = \begin{bmatrix}0 & +1 & 0\\+1 & -4 & +1\\0 & +1 & 0\end{bmatrix}$$
# 
# For taking diagonals into consideration,
# $$kernel = \begin{bmatrix}+1 & +1 & +1\\+1 & -8 & +1\\+1 & +1 & +1\end{bmatrix}$$
# <img src="../resources/laplacian.jpg">
# Function: <b>cv2.Laplacian(image, depth[, dst[, ksize...]])</b>
# 

laplacian = cv2.Laplacian(blurred, cv2.CV_16U)

plt.imshow(laplacian, cmap='gray')
plt.show()


# ## Canny Edge Detection
# Apart from first order derivatives, this also utilizes non-maxima supression and hysteresis thresholding in order to detect weak and strong edges. It uses a multi-stage algorithm to detect a wide range of edges in images.
# 
# <b>Process</b>:
# 1. <u>Noise Supression</u> - To smooth the image and reduce noise, a Gaussian filter is applied to convolve with the image. The larger the size of kernel, the lower the detector’s sensitivity to noise. A 5×5 is a good size for most cases.
# 
# 2. <u>Finding Intensity Gradient of Image</u> - Uses four filters to detect horizontal, vertical and diagonal edges in the blurred image(Sobel kernel is preferred). These return first order derivatives in horizontal direction (G<sub>x</sub>) and vertical direction (G<sub>y</sub>)
# $$G = \sqrt{G_{x}^2 + G_{y}^2}$$
# $$\theta = \tan^{-1}\frac{G_{y}}{G_{x}}$$
# 
# 3. <u>Non Maxima Supression</u> - It's an edge thinning technique. At every pixel, pixel is checked if it is a local maximum in its neighborhood in the direction of gradient. Thus, all the gradient values are supressed to 0 except the local maximal, which indicates location with the sharpest change of intensity value.
#     - Compare the edge strength of the current pixel with the edge strength of the pixel in the positive and negative gradient directions.
#     - If the edge strength of the current pixel is the largest compared to the other pixels in the mask with the same direction, the value will be preserved. Otherwise, the value will be suppressed.
# 4. <u>Double Thresholding</u> - Some edge pixels due to noise and color variation are still left. Two threshold values are selected => maxThresh and minThresh. Pixels with value above maxThresh are labelled strong edges, those below minThresh are supressed and those lying between minThresh and maxThresh are labelled weak edges.
# <img src="../resources/canny.jpg">
# 5. <u>Hysteresis Edge Tracking</u> - Connected component analysis is done on all the weak edges now. Lets say our hysteresis threshold is 3. So, if a weak edge pixel has 3 or more than 3 strong edge pixels amongst its 8 neighboring pixels, it will be marked as an edge, otherwise will be supressed.
# 

canny_inbuilt = cv2.Canny(gray, 100, 200)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(canny_inbuilt, cmap='gray')

def canny_experimental(image):
    # Noise Supression
    sigma = 1.4  # experimental
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    blurred = np.asarray(blurred, dtype="int32")
    
    # Intensity Gradient
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobelx = ndimage.convolve(blurred, kernel_x)
    sobely = ndimage.convolve(blurred, kernel_y)
    sobel = np.hypot(sobelx, sobely)
    
    ## Approx directions to horizontal, vertical, left diagonal and right diagonal
    sobel_dir = np.arctan2(sobely, sobelx)
    for x in range(sobel_dir.shape[0]):
        for y in range(sobel_dir.shape[1]):
            dir = sobel_dir[x][y]
            if 0<= dir < 22.5 or 157.5 <= dir <= 202.5 or 337.5 <= dir <= 360:
                sobel_dir[x][y] = 0  # horizontal
            elif 22.5 <= dir < 67.5 or 202.5 <= dir < 247.5:
                sobel_dir[x][y] = 45  # left diagonal
            elif 67.5 <= dir < 112.5 or 247.5 <= dir < 292.5:
                sobel_dir[x][y] = 90  # vertical
            else:
                sobel_dir[x][y] = 135  # right diagonal
    
    # Non-maxima Supression
    sobel_magnitude = np.copy(sobel)
    for x in range(1, sobel.shape[0] - 1):
        for y in range(1, sobel.shape[1] - 1):
            # Compare magnitude of gradients in the direction of gradient of pixel
            # If value is less than any of the neighoring magnitude, make pixel 0
            # We are checking in 3x3 neighbourhood
            if sobel_dir[x][y] == 0 and sobel[x][y] <= min(sobel[x][y+1], sobel[x][y-1]):
                    sobel_magnitude[x][y] = 0
            elif sobel_dir[x][y] == 45 and sobel[x][y] <= min(sobel[x-1][y+1], sobel[x+1][y-1]):
                sobel_magnitude[x][y] = 0
            elif sobel_dir[x][y] == 90 and sobel[x][y] <= min(sobel[x-1][y], sobel[x+1][y]):
                sobel_magnitude[x][y] = 0
            elif sobel[x][y] <= min(sobel[x+1][y+1], sobel[x-1][y-1]):
                sobel_magnitude[x][y] = 0
    
    # Double Thresholding
    sobel = sobel_magnitude
    canny = np.zeros_like(sobel)
    strong_edge = np.zeros_like(sobel)
    weak_edge = np.zeros_like(sobel)
    thresh = np.max(sobel)
    maxThresh = 0.2 * thresh
    minThresh = 0.1 * thresh
    for i in range(sobel.shape[0]):
        for j in range(sobel.shape[1]):
            if sobel[i][j] >= maxThresh:
                canny[i][j] = sobel[i][j]
                strong_edge[i][j] = sobel[i][j]
                weak_edge[i][j] = 0
            elif sobel[i][j] >= minThresh:
                canny[i][j] = sobel[i][j]
                weak_edge[i][j] = sobel[i][j]
                strong_edge[i][j] = 0
            else:
                canny[i][j] = 0
                strong_edge[i][j] = 0
                weak_edge[i][j] = 0
    
    # Connected Component Analysis
    neighbor_thresh = 2
    for i in range(weak_edge.shape[0]):
        for j in range(weak_edge.shape[1]):
            neighbors = 0
            if weak_edge[i][j] == 0:
                continue
            # check for corner
            if i == 0 and j == 0:
                if strong_edge[1][0] != 0:       neighbors += 1
                if strong_edge[1][1] != 0:       neighbors += 1
                if strong_edge[0][1] != 0:       neighbors += 1
            elif i == weak_edge.shape[0] - 1 and j == 0:
                if strong_edge[i-1][0] != 0:     neighbors += 1
                if strong_edge[i-1][1] != 0:     neighbors += 1
                if strong_edge[i][1] != 0:       neighbors += 1
            elif i == 0 and j == weak_edge.shape[1] - 1:
                if strong_edge[i][j-1] != 0:     neighbors += 1
                if strong_edge[i+1][j-1] != 0:   neighbors += 1
                if strong_edge[i+1][j] != 0:     neighbors += 1
            elif i == weak_edge.shape[0] - 1 and j == weak_edge.shape[1] - 1:
                if strong_edge[i-1][j] != 0:     neighbors += 1
                if strong_edge[i-1][j-1] != 0:   neighbors += 1
                if strong_edge[i][j-1] != 0:     strong_edge += 1
            # check for edge
            elif i == 0:
                if strong_edge[i][j-1] != 0:     neighbors += 1
                if strong_edge[i+1][j-1] != 0:   neighbors += 1
                if strong_edge[i+1][j] != 0:     neighbors += 1
                if strong_edge[i+1][j+1] != 0:   neighbors += 1
                if strong_edge[i][j+1] != 0:     neighbors += 1
            elif i == weak_edge.shape[0] - 1:
                if strong_edge[i][j-1] != 0:     neighbors += 1
                if strong_edge[i-1][j-1] != 0:   neighbors += 1
                if strong_edge[i-1][j] != 0:     neighbors += 1
                if strong_edge[i-1][j+1] != 0:   neighbors += 1
                if strong_edge[i][j+1] != 0:     neighbors += 1
            elif j == 0:
                if strong_edge[i-1][j] != 0:     neighbors += 1
                if strong_edge[i-1][j+1] != 0:   neighbors += 1
                if strong_edge[i][j+1] != 0:     neighbors += 1
                if strong_edge[i+1][j+1] != 0:   neighbors += 1
                if strong_edge[i+1][j] != 0:     neighbors += 1
            elif j == weak_edge.shape[1] - 1:
                if strong_edge[i-1][j] != 0:     neighbors += 1
                if strong_edge[i-1][j-1] != 0:   neighbors += 1
                if strong_edge[i][j-1] != 0:     neighbors += 1
                if strong_edge[i+1][j-1] != 0:   neighbors += 1
                if strong_edge[i+1][j] != 0:     neighbors += 1
            # check for the 8 neighboring strong edge pixels
            else:
                if strong_edge[i-1][j-1] != 0:   neighbors += 1
                if strong_edge[i-1][j] != 0:     neighbors += 1
                if strong_edge[i-1][j+1] != 0:   neighbors += 1
                if strong_edge[i][j+1] != 0:     neighbors += 1
                if strong_edge[i+1][j+1] != 0:   neighbors += 1
                if strong_edge[i+1][j] != 0:     neighbors += 1
                if strong_edge[i+1][j-1] != 0:   neighbors += 1
                if strong_edge[i][j-1] != 0:     neighbors += 1
            # supress if no strong edges in neihborhood
            if neighbors < neighbor_thresh: canny[i][j] = 0
        
    canny = cv2.erode(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)), iterations=1)
    
    return canny

canny_implemented = canny_experimental(gray)
fig.add_subplot(1, 2, 2)
plt.imshow(canny_implemented, cmap='gray')

plt.show()


import cv2
import numpy as np
from matplotlib import pyplot as plt


# ## Drawing stuff on an image
# 1. <b>line(image, point1, point2, color, thickness)</b>
# 2. <b>rectangle(image, point1, point2, color, thickness)</b>
# 3. <b>circle(image, centre, radius, color, thickness)</b>
# 4. <b>ellipse(image, centre, axes, angle, start_angle, end_angle, color, thickness)</b>
# 

# Matplotlib uses RGB whereas CV uses BGR
def inverted(image):
    inv_image = np.zeros_like(image)
    inv_image[:,:,0] = image[:,:,2]
    inv_image[:,:,1] = image[:,:,1]
    inv_image[:,:,2] = image[:,:,0]
    return inv_image


image = cv2.imread("../resources/messi.jpg")

# Draw a line
cv2.line(image, (40, 40), (100, 200), (255, 0, 0), 2)

# Draw a rectangle
cv2.rectangle(image, (300, 300), (200, 250), (0, 0, 255), 3)

# Draw a circle
cv2.circle(image, (100, 200), 50, (0, 255, 0), 4)

# Draw an ellipse
cv2.ellipse(image, (300, 100), (100, 80), 45, 0, 360, (255, 255, 255), 5)

plt.imshow(inverted(image))
plt.show()


# Drawing a polygon

image = cv2.imread("../resources/messi.jpg")

points = np.array([
    [10,10],
    [100, 200],
    [200, 300],
    [300, 200],
    [300, 10]
])

cv2.polylines(image, [points], True, (0, 255, 0), 4)

plt.imshow(inverted(image))
plt.show()


# Filling the polygon
cv2.fillConvexPoly(image, points, (100, 100, 100))

plt.imshow(inverted(image))
plt.show()


# Filling multiple polygons
image = cv2.imread("../resources/messi.jpg")

triangle = np.array([
    [10, 10],
    [10, 50],
    [60, 10]
])

square = np.array([
    [100, 100],
    [100, 200],
    [200, 200],
    [200, 100]
])

cv2.fillPoly(image, [triangle, square], (200, 200, 0))

# Writing text
cv2.putText(image, "MESSI", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

plt.imshow(inverted(image))
plt.show()


# ## Sample code to draw on image with mouse clicks
# 

image = cv2.imread("../resources/messi.jpg")

# flags and params will not be used, they are passed by opencv itself, if any
def mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 10, (200, 200, 0), 3)


cv2.namedWindow('testing')
cv2.setMouseCallback('testing', mouse_click)
        
while True:
    cv2.imshow('testing', image)
    k = cv2.waitKey(10) & 255
    if k == 27:
        cv2.destroyWindow('testing')
        break

plt.imshow(inverted(image))
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt


# # Fourier Transform
# Fourier Transform is used to decompose an image into its sine and cosine components. The output of the transformation represents the image in the Fourier or frequency domain, while the input image is the spatial domain equivalent. In the Fourier domain image, each point represents a particular frequency contained in the spatial domain image. It's used for image filtering, compression etc.
# 
# We mostly use <b>DFT</b> (Discrete Fourier Transform), which is sample fourier transform and thus, doesn't contain all the frequencies, but only a set of samples which is large enough to describe the spatial domain information.
# 
# For an image of nxm dimensions, 2-D DFT is given by
# $$F(k, l) = \sum_{i=0}^{n-1}\sum_{j=0}^{m-1}f(i, j)e^{-i2\pi(\frac{ki}{n} + \frac{lj}{m})}$$
# 
# Inverse transform is given by
# $$f(a, b) = \frac{1}{n*m}\sum_{k=0}^{n-1}\sum_{l=0}^{m-1}F(k, l)e^{i2\pi(\frac{ka}{n} + \frac{lb}{m})}$$
# 

image = cv2.imread("../resources/messi.jpg", cv2.IMREAD_GRAYSCALE)

fig = plt.figure()
fig.set_size_inches(18, 10)

dft = np.fft.fftshift(cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT))
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft[:,:,0], dft[:,:,1]))
# phase_spectrum = cv2.phase(dft[:,:,0], dft[:,:,1])
magnitude_spectrum, phase_spectrum = cv2.cartToPolar(dft[:,:,0], dft[:,:,1])
magnitude_spectrum = 20*np.log(magnitude_spectrum)

fig.add_subplot(3, 2, 1)
plt.imshow(magnitude_spectrum, cmap='gray')
fig.add_subplot(3, 2, 2)
plt.imshow(phase_spectrum, cmap='gray')

# Applying Low Pass Filter => Removes high frequencies (noise)
rows, cols = image.shape
center = rows/2, cols/2
mask = np.zeros((rows, cols, 2), np.uint8)
# low pass filter => 1 at low frequency and 0 at high frequency
mask[center[0] - 30:center[0] + 30, center[1] - 30:center[1] + 30] = 1

filtered = dft*mask
inv = cv2.idft(np.fft.ifftshift(filtered))
inv_mag = cv2.magnitude(inv[:,:,0], inv[:,:,1])
fig.add_subplot(3, 2, 3)
plt.imshow(image, cmap='gray')
fig.add_subplot(3, 2, 4)
plt.imshow(inv_mag, cmap='gray')

# Applying high pass filter
dft[center[0] - 30:center[0] + 30, center[1] - 30:center[1] + 30] = 0
inv = cv2.idft(np.fft.ifftshift(dft))
inv_mag = cv2.magnitude(inv[:,:,0], inv[:,:,1])
fig.add_subplot(3, 2, 5)
plt.imshow(image, cmap='gray')
fig.add_subplot(3, 2, 6)
plt.imshow(inv_mag, cmap='gray')

plt.show()


# Representation of some important kernels

gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

fig = plt.figure()
fig.set_size_inches(18, 10)

kernels = [sobel_x, sobel_y, gauss, laplacian]
for i in range(4):
    dft = np.fft.fftshift(cv2.dft(np.float32(kernels[i]), flags=cv2.DFT_COMPLEX_OUTPUT))
    magnitude = np.log(cv2.magnitude(dft[:,:,0], dft[:,:,1]) + 1)
    fig.add_subplot(2, 2, i+1)
    plt.imshow(magnitude, cmap='gray')

plt.show()


# ## Applications of Fourier Transforms
# 1. Foresnic Image Analysis
# <img src="../resources/fourier1.png">
# 2. Removing Fluctuations
# <img src="../resources/fourier2.png">
# 

# # Hough Transform
# It's a feature extraction technique. The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure.
# 

# Hough Line Transform

image = cv2.imread("../resources/sudoku1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
copy = image.copy()

fig = plt.figure()
fig.set_size_inches(18, 10)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
for line in lines:
    for radius, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * radius
        y0 = b * radius
        x1 = int(x0 + 1000 * (-b))
        x2 = int(x0 - 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        y2 = int(y0 - 1000 * a)

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

fig.add_subplot(1, 2, 1)
plt.imshow(image)

edges = cv2.Canny(gray, 50, 150)
min_length = 90
max_gap = 18
lines = cv2.HoughLinesP(edges, 1, np.pi/180, min_length, max_gap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

fig.add_subplot(1, 2, 2)
plt.imshow(copy)
plt.show()


# Hough Circle Transform
image = cv2.imread("../resources/template_test.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 100, 200)

try:
    circles = cv2.HoughCircles(edges, cv2.cv.CV_HOUGH_GRADIENT, 1, 10, param1=200, param2=90)
except:
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10, param1=200, param2=90)
circles = np.uint(np.around(circles))

for circle in circles:
    for center_x, center_y, radius in circle:
        cv2.circle(image, (center_x, center_y), radius, (255, 0, 0), 3)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()


# # Template Matching
# Searching and finding the location of a template image in a larger image.
# 
# Matching Methods Available:
# 1. cv2.TM_SQDIFF
# 2. cv2.TM_SQDIFF_NORMED
# 3. cv2.TM_CCORR
# 4. cv2.TM_CCORR_NORMED
# 5. cv2.TM_CCOEFF
# 6. cv2.TM_CCOEFF_NORMED
# 

image = cv2.imread("../resources/template_test.png")
template = cv2.imread("../resources/emoji.png")
template = cv2.resize(template, (80, 80))
w, h = template.shape[:2][::-1]

fig = plt.figure()
fig.set_size_inches(18, 10)

for method in range(6):
    copy = image.copy()
    match = cv2.matchTemplate(copy, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(copy, top_left, bottom_right, (255, 0, 0), 2)
    fig.add_subplot(2, 3, method+1)
    plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
plt.show()


# ## Distance Transform
# An operator that transforms the values of pixels in an image according to their distance from the boundary of the object. Farther the pixel from the boundary, higher the value it gets. This results in a shrinked image (boundary pixels -> black, pixels near center -> white).
# 
# Applications:
# - Generating skeleton of images (connectivity, length and width).
# - Navigation and pathfinding
# 

image = cv2.imread("../resources/cells.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
# plt.imshow(thresh, cmap='gray')
# plt.show()
try:
    distance = cv2.distanceTransform(thresh, cv2.cv.CV_DIST_L2, 5)
except:
    distance = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
plt.imshow(distance, cmap='gray')
plt.show()


# # Watershed Transformation
# The term watershed refers to a ridge that divides areas drained by different river systems. Computer analysis of image objects starts with finding them-deciding which pixels belong to each object. This is called image segmentation, the process of separating objects from the background, as well as from each other.
# 
# Understanding the watershed transform requires that you think of an image(grayscale) as a topographic surface where high intensity denotes peaks and low intensity denotes valleys Every local minima is filled with different color.
# 
# Things we should know before continuing:
# 1. Thresholding
# 2. Morphological Operations
# 3. Distance Transform
# 
# Applications:
# - Used to monitor traffic. It automatically segments the lanes of a road to count the number of vehicles on different lanes.
# - Detecting fractures in surface of steel.
# 

image = cv2.imread("../resources/watershed_test.jpg")
# Segmentation needs a grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image, contours, -1, (0, 0, 255), 1)

kernel = np.ones((3, 3))
# opening is done to remove small noises
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# controlling dilation iteration provides us with region where we are sure that object doesn't exists.
background = cv2.dilate(opening, kernel, iterations=2)

# Approximate the actual object regions using distance transform
try:
    dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
except:
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, foreground = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, cv2.THRESH_BINARY)

background = np.uint8(background)
foreground = np.uint8(foreground)

# We have background and foreground, now we need to find whether remaining region belongs to our object or not.
# These regions are borders (common to both foreground and background)
unknown = cv2.subtract(background, foreground)

# We mark the sure regions with incrementing labels, and the unknown regions with 0
# For opencv 2.x.x or above
####
contours, hierarchy = cv2.findContours(foreground, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
markers = np.zeros(gray.shape, dtype=np.int32)
markers = np.int32(background) + np.int32(foreground)
for idx in range(len(contours)):
    cv2.drawContours(markers, contours, idx, idx+2, -1)
####
# For opencv 3.x.x or above
####
# ret, markers = cv2.connectedComponents(foreground)
####
markers += 1
markers[unknown == 255] = 0

fig = plt.figure()
fig.set_size_inches(15, 8)

fig.add_subplot(1, 3, 1)
plt.imshow(markers)

cv2.watershed(image, markers)
image[markers == -1] = (0, 0, 255)

fig.add_subplot(1, 3, 2)
plt.imshow(markers)
plt.colorbar()
fig.add_subplot(1, 3, 3)
plt.imshow(gray, cmap='gray')
plt.show()





import cv2
import numpy as np
from matplotlib import pyplot as plt


# ## Histograms
# Image histogram provides a plot for intensity distribution of an image, with pixel values (0-255) on x-axis, and corresponding number of pixels on y-axis.
# 
# ### Finding histogram of an image
# Function: <b>cv2.calcHist(images, channels, mask, histSize, ranges)</b>
# 1. <b>images</b> - source image (datatype - uint8 or float32)
# 2. <b>channels</b> - Channel for which histogram needs to be calculated
# 3. <b>mask</b> - For finding histogram for a region of image, we have to create a mask. Otherwise "None".
# 4. <b>histSize</b> - Count of intervals for finding pixel values.
# 5. <b>ranges</b> - Range of intensity values to be measured.
# 

image = cv2.imread('../resources/messi.jpg')

colors = ['blue', 'green', 'red']

fig = plt.figure()
fig.set_size_inches(18, 5)

for i in range(len(colors)):
    histogram = cv2.calcHist([image], [i], None, [256], [0,256])
    fig.add_subplot(1, 1, 1)
    plt.plot(histogram, color=colors[i])
    plt.xlim([0, 256])

plt.show()


# ### Histogram Equalisation
# Application - Enhancing contrast of an image
# 
# Function: <b>cv2.equalizeHist(image)</b>
# 

# Contrasting grayscale image
gray_scene = cv2.imread('../resources/scene_gray.png', cv2.IMREAD_GRAYSCALE)
histogram = cv2.equalizeHist(gray_scene)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(gray_scene, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(histogram, cmap='gray')

plt.show()


# Contrasting color image
# Concept of histogram equalization is applicable to intensity values only.
# So we convert BGR to YUV colorspace, separating color information from intensity.
scene = cv2.imread('../resources/scene.png')
yuv_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2YUV)

y_channel = yuv_scene[:, :, 0]
y_channel_equalized = cv2.equalizeHist(y_channel)
yuv_scene[:, :, 0] = y_channel_equalized

contrasted_scene = cv2.cvtColor(yuv_scene, cv2.COLOR_YUV2BGR)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(contrasted_scene, cv2.COLOR_BGR2RGB))

plt.show()


# ### CLAHE (Contrast Limited Adaptive Histogram Equalization)
# Equalizing the historgram across whole image might not be useful sometimes. Small objects in the image might lose information, due to over or under contrasting. In this case, adaptive histogram equalization is used.
# 
# In CLAHE, image is divided into small blocks (8x8 by default), and these blocks are histogram equalized rather than whole image once.
# 
# To avoid amplification of noise, contrast limit (40 by default) is applied. If any histogram bin is above contrast limit, it is clipped and distributed among other bins, and then histogram is equalized.
# 
# Tile boundary pixels, corner pixels and edge pixels might be transformed more than one time, thus giving multiple values for the same pixel. For simplicity and computational efficiency, certain interpolation methods are used to calculate the pixel value from the multiple obtained values.
# 

statue = cv2.imread('../resources/statue.jpg', cv2.IMREAD_GRAYSCALE)

normal_equalized_hist = cv2.equalizeHist(statue)

clahe = cv2.createCLAHE()
clahe_equalized_hist = clahe.apply(statue)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 3, 1)
plt.imshow(statue, cmap='gray')

fig.add_subplot(1, 3, 2)
plt.imshow(normal_equalized_hist, cmap='gray')

fig.add_subplot(1, 3, 3)
plt.imshow(clahe_equalized_hist, cmap='gray')

plt.show()


# ### Histogram Backprojection Example
# We'll be finding the football in the picture below.
# <img src="../resources/messi.jpg">
# Our goal is to create an output image where each pixel corresponds to probability of pixel belonging to the object.
# 

target = image
hsv_target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

football = target[290:335, 338:387]
hsv_obj = cv2.cvtColor(football, cv2.COLOR_BGR2HSV)

obj_hist = cv2.calcHist([hsv_obj], [0, 1], None, [40, 256], [0, 40, 0 ,256])

cv2.normalize(obj_hist, obj_hist, 0, 255, cv2.NORM_MINMAX)
result = cv2.calcBackProject([hsv_target], [0, 1], obj_hist, [0, 40, 0, 256], 1)

plt.imshow(result, cmap='gray')
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt


# # SIFT (Surface Invariant Feature Transform)
# Previously discussed corner detectors (Harris and Shi-Tomasi) were rotation invariant. But when it comes to images of different scales, matching features becomes a common problem.
# 
# <center>Normal Detectors<img src="../resources/scaling.jpg"></center>
# 
# SIFT can work even after changing following parameters(other than scale. duh!):
# - Rotation
# - Illumination
# - Viewpoint
# 
# ### Breaking down and understanding the algorithm
# #### 1. Constructing a scale space
# You take the original image, and generate progressively blurred out images. Then, you resize the original image to half size. And you generate blurred out images again. And you keep repeating.
# <img src="../resources/octave.jpg">
# We generate several octaves of the original image. Each octave's image is half the size of previous one. In each octave, images are progressively blurred using the Gaussian Blur operator. Generally, 4 octaves and 5 blur levels are considered ideal for the algorithm.
# #### 2. LoG (Laplacian of Gaussian) Approximations
# From color space, we take images, blur them and calculate second order derivatives (Laplacian), which provides corners and edges. As second order derivatives are extremely sensitive to noise, it's necessary to apply blur in order to supress noise.
# 
# LOG is computationally heavy, so we opt for DoG (Difference of Gaussian), which produces approximately equivalent outputs as LoG, and is faster (less computations).
# 
# Two consecutive images in an octave are picked and subtracted. Then the next consecutive pair is taken, and the process repeats. This is done for all octaves. The resulting images are an approximation of scale invariant LoG (which is good for detecting keypoints).
# 
# $$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$
# 
# $$\sigma\text{ is the scale parameter. More is it's value, more is the blur}$$
# 
# $$\text{If amount of blur in an image of an octave is }\sigma\text{, then amount of blur for next image in the same octave will be }k\sigma\text{, where k is any constant}$$
# 
# $$\text{Across octaves, }\sigma\text{ varies, and withing octaves, constant 'k' varies}$$
# <img src="../resources/blur_amount.jpg">
# #### 3. Keypoints Localization in DoG images
# <img src="../resources/sift_local_extrema.jpg">
# Each pixel is compared to it's 8 neighbours and 3x3 corresponding windows in image above and below it. Suppose, X is the pixel being compared here. It's marked as a keypoint if it's greatest or least among the 26(8 + 3x3 + 3x3) neighbours. Now, the approximate maxima and minima are located. To find accurate positions, we apply sub pixel accuracy method.
# 
# Using the available pixel data, subpixel values are generated. This is done by the Taylor expansion of the image around the approximate key point. We just need to find extreme points of the equation:
# $$D(X) = D + (\frac{\delta D}{\delta X})^{T}X + \frac{1}{2}X^T\frac{\delta^{2}D}{\delta X^2}X$$
# $$\text{D(X) is scale-space function, X = }(x, y, \sigma)^T$$
# #### 4. Refining Keypoints
# Remember that DoG contains not only corners but also edges. So, now we'll reject the low contrast and edge region keypoints.
# 
# <u>Removing low contrast keypoints</u>: If magnitude of current pixel in DoG image is less than a threshold, it's rejected. After this, we use taylor expansion to get intensity value at sub-pixel locations. If this is less than the threshold, the keypoint is rejected.
# 
# <u>Removing edge region keypoints</u>: A 2x2 Hessian matrix is used to calculate principal curvature, which is then used to find eigen values for the shape operator. We know that for edges, one eigen value is greater than the other. This way, edge region keypoinds are discarded as well.
# #### 5. Rotation Invariance
# A neighbourhood is taken around a keypoint depending on the scale, and gradient magnitude and direction is calculated in that region.
# $$m(x, y) = \sqrt{(L(x+1, y) - L(x-1, y))^2 + (L(x, y+1) - L(x, y-1))^2}$$
# $$\theta(x, y) = \tan^{-1}\frac{L(x, y+1) - L(x, y-1)}{L(x+1, y) - L(x-1, y)}$$
# A histogram is made in which, the 360 degrees of orientation are broken into 36 bins (each of 10 degrees). The keypoint is assigned the index of the bin with maximum peak in the histogram. Also, all the other peaks above 80% of the maximum peak are converted to a new keypoint.
# #### 6. Keypoint Descriptor
# A 16x16 neigbourhood around the keypoint is taken, and is divided into 16 4x4 sub-blocks. For each 4x4 block, gradient magnitudes and orientations are calculated. These orientations are put into an 8 bin histogram. A gaussian weighting function is used to alter the orientation magnitudes before feeding them into the histogram. So total of 128 (4*4*8) bin values are available for each keypoint, which are then normalized. These numbers form "feature vector" or "feature descriptor", and it's unique for every keypoint.
# #### 7. Final Touch
# As feature vetors depend on gradient orientations, rotating the image will change the vector as well. But we want it to be rotation invariant. So, keypoint's orientation is subtracted from each orienation to make it each gradient orientation realtive to the orientation of the keypoint. Thresholding will provide the illumination invariance.
# 

image = cv2.imread("../resources/messi.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

try:
    sift = cv2.SIFT()
except:
    sift = cv2.xfeatures2d.SIFT_create()

kp, descriptors = sift.detectAndCompute(gray, mask=None)
# print descriptors
# print len(kp), descriptors.shape

flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT
try:
    image = cv2.drawKeypoints(gray, kp, flags=flags)
except:
    cv2.drawKeypoints(gray, kp, image, flags=flags)

fig = plt.figure()
fig.set_size_inches(10, 10)
fig.add_subplot(1,1,1)
plt.imshow(image)
plt.show()


# Let's get our hands dirty :D
# Now we'll try to implement our own SIFT
import math
import random


class Descriptor(object):
    def __init__(self, x, y, feature_vector):
        self.x = x
        self.y = y
        self.feature_vector = feature_vector


class Keypoint(object):
    def __init__(self, x, y, magnitude, orientation, scale):
        self.x = x
        self.y = y
        self.magnitude = magnitude
        self.orientation = orientation
        self.scale = scale


class SIFT(object):
    def __init__(self, image, octaves, intervals):
        # CONSTANTS
        self.sigma_antialias = 0.5
        self.sigma_preblur = 1.0
        self.edge_threshold = 7.2
        self.intensity_threshold = 0.05
        self.pi = 3.1415926535897932384626433832795
        self.bins = 36
        self.max_kernel_dim = 20
        self.feature_win_dim = 16
        self.descriptor_bins = 8
        self.feature_vector_size = 128
        self.feature_vector_threshold = 0.2
        self.num_keypoints = 0
        self.keypoints = []
        self.descriptors = []
        
        self.image = image.copy()  # colored image (BGR format)
        self.octaves = octaves
        self.intervals = intervals
        
        self.gaussian = [[None for interval in range(self.intervals + 3)] for octave in range(self.octaves)]
        self.dog = [[None for interval in range(self.intervals + 2)] for octave in range(self.octaves)]
        self.extrema = [[None for interval in range(self.intervals)] for octave in range(self.octaves)]
        self.sigma_level =[[None for interval in range(self.intervals + 3)] for octave in range(self.octaves)]
        
    def build_scale_space(self):
        gray = np.float32(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
        
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                gray[i, j] /= 255.0
        
        ksize = int(3 * self.sigma_antialias)
        if ksize % 2 == 0: ksize += 1
        gray = cv2.GaussianBlur(gray, (ksize, ksize), self.sigma_antialias)
        
        self.gaussian[0][0] = np.float32(cv2.pyrUp(gray))
        ksize = int(3 * self.sigma_preblur)
        if ksize % 2 == 0: ksize += 1
        self.gaussian[0][0] = np.float32(
            cv2.GaussianBlur(self.gaussian[0][0], (ksize, ksize), self.sigma_preblur)
        )
        
        sigma_init = math.sqrt(2.0)
        self.sigma_level[0][0] = sigma_init * 0.5
        
        for i in range(self.octaves):
            sigma = sigma_init
            
            for j in range(1, self.intervals + 3):
                sigma_next = math.sqrt((2 ** (2.0 / self.intervals)) - 1) * sigma
                sigma *= 2 ** (1.0 / self.intervals)
                
                self.sigma_level[i][j] = sigma * 0.5 * (2 ** i)
                ksize = int(3 * sigma_next)
                if ksize % 2 == 0: ksize += 1
                
                self.gaussian[i][j] = np.float32(
                    cv2.GaussianBlur(self.gaussian[i][j - 1], (ksize, ksize), sigma_next)
                )
                
                self.dog[i][j - 1] = np.float32(cv2.subtract(self.gaussian[i][j - 1], self.gaussian[i][j]))
            
            if i != self.octaves - 1:
                self.gaussian[i + 1][0] = np.float32(cv2.pyrDown(self.gaussian[i][0]))
                self.sigma_level[i + 1][0] = self.sigma_level[i][self.intervals]
    
    def detect_extrema(self):
        num_keypoints = 0
        
        for i in range(self.octaves):
            for j in range(1, self.intervals + 1):
                self.extrema[i][j - 1] = np.float32(np.zeros_like(self.dog[i][0]))
                
                middle = self.dog[i][j]
                above = self.dog[i][j + 1]
                below = self.dog[i][j - 1]
                
                for x in range(1, self.dog[i][j].shape[0] - 1):
                    for y in range(1, self.dog[i][j].shape[1] - 1):
                        flag = False
                        
                        pixel = middle[x, y]
                        
                        # Sigh! Have to check against 26 pixel if you remember (9 + 9 + 8)
                        values = [
                            middle[x - 1, y - 1], middle[x, y - 1], middle[x + 1, y - 1],
                            middle[x - 1, y], middle[x, y], middle[x + 1, y],
                            middle[x - 1, y + 1], middle[x, y + 1], middle[x + 1, y + 1],
                            
                            above[x - 1, y - 1], above[x, y - 1], above[x + 1, y - 1],
                            above[x - 1, y], above[x, y], above[x + 1, y],
                            above[x - 1, y + 1], above[x, y + 1], above[x + 1, y + 1],
                            
                            below[x - 1, y - 1], below[x, y - 1], below[x + 1, y - 1],
                            below[x - 1, y], below[x, y], below[x + 1, y],
                            below[x - 1, y + 1], below[x, y + 1], below[x + 1, y + 1]
                        ]
                        values.sort()
                        
                        # Check for maxmima
                        if values[-1] == pixel and values[-1] != values[-2]:
                            flag = True
                            self.extrema[i][j - 1][x, y] = 255
                            num_keypoints += 1
                        # Check for minima
                        elif values[0] == pixel and values[0] != values[1]:
                            flag = True
                            self.extrema[i][j - 1][x, y] = 255
                            num_keypoints += 1
                        
                        # Intensity check
                        if flag and math.fabs(middle[x, y]) < self.intensity_threshold:
                            self.extrema[i][j - 1][x, y] = 0
                            num_keypoints -= 1
                            flag = False
                        
                        # Edge check
                        if flag:
                            # Using Hessian Matrix
                            dx2 = middle[x, y - 1] + middle[x, y + 1] - 2 * middle[x, y]
                            dy2 = middle[x - 1, y] + middle[x + 1, y] - 2 * middle[x, y]
                            dxy = (
                                middle[x - 1, y - 1] + 
                                middle[x - 1, y + 1] + 
                                middle[x + 1, y - 1] + 
                                middle[x + 1, y + 1]
                            )
                            dxy /= 4.0
                            
                            tr = dx2 + dy2
                            det = dx2 * dy2 + dxy ** 2
                            
                            curvature_ratio = (tr ** 2) / det
                            if det < 0 or curvature_ratio > self.edge_threshold:
                                self.extrema[i][j - 1][x, y] = 0
                                num_keypoints -= 1

        self.num_keypoints = num_keypoints
    
    def assign_orientation(self):
        magnitudes = [[None for interval in range(self.intervals)] for octave in range(self.octaves)]
        orientations = [[None for interval in range(self.intervals)] for octave in range(self.octaves)]
        
        for i in range(self.octaves):
            for j in range(1, self.intervals):
                magnitudes[i][j - 1] = np.float32(np.zeros_like(self.gaussian[i][j]))
                orientations[i][j - 1] = np.float32(np.zeros_like(self.gaussian[i][j]))
                
                for x in range(1, self.gaussian[i][j].shape[0] - 1):
                    for y in range(1, self.gaussian[i][j].shape[1] - 1):
                        # Calculate gradient
                        dx = self.gaussian[i][j][x + 1, y] - self.gaussian[i][j][x - 1, y]
                        dy = self.gaussian[i][j][x, y + 1] - self.gaussian[i][j][x, y - 1]
                        
                        magnitudes[i][j - 1][x, y] = math.sqrt(dx ** 2 + dy ** 2)
                        orientations[i][j - 1][x, y] = math.atan2(dy, dx)
        
        for i in range(self.octaves):
            scale = 2.0 ** i
            
            for j in range(1, self.intervals):
                sigma = self.sigma_level[i][j]
                
                ksize = int(3 * 1.5 * sigma)
                if ksize % 2 == 0: ksize += 1
                
                weighted = np.float32(cv2.GaussianBlur(magnitudes[i][j - 1], (ksize, ksize), 1.5 * sigma))
                approx_gaussian_kernel_size = ksize / 2
                
                mask = np.float32(np.zeros_like(self.gaussian[i][0]))
                
                for x in range(self.gaussian[i][0].shape[0]):
                    for y in range(self.gaussian[i][0].shape[1]):
                        if self.extrema[i][j - 1][x, y] != 0:
                            orientation_hist = [0.0 for b in range(self.bins)]
                            
                            for ii in range(-approx_gaussian_kernel_size, approx_gaussian_kernel_size + 1):
                                for jj in range(-approx_gaussian_kernel_size, approx_gaussian_kernel_size + 1):
                                    if x + ii < 0 or x + ii >= self.gaussian[i][0].shape[0]: continue
                                    if y + jj < 0 or y + jj >= self.gaussian[i][0].shape[1]: continue
                                    
                                    sampled_orientation = orientations[i][j - 1][x + ii, y + jj]
                                    sampled_orientation += self.pi
                                    
                                    degrees = sampled_orientation * 180 / self.pi
                                    orientation_hist[int(degrees * self.bins / 360)] += weighted[x + ii, y + jj]
                                    
                                    mask[x + ii, y + jj] = 255
                            
                            max_peak = max(orientation_hist)
                            max_peak_index = orientation_hist.index(max_peak)
                            
                            o = []
                            m = []
                            
                            for k in range(self.bins):
                                if orientation_hist[k] > 0.8 * max_peak:
                                    x1 = k - 1
                                    x2 = k
                                    y2 = orientation_hist[k]
                                    x3 = k + 1
                                    
                                    if k == 0:
                                        y1 = orientation_hist[self.bins - 1]
                                        y3 = orientation_hist[k + 1]
                                    elif k == self.bins - 1:
                                        y1 = orientation_hist[k - 1]
                                        y3 = orientation_hist[0]
                                    else:
                                        y1 = orientation_hist[k - 1]
                                        y3 = orientation_hist[k + 1]
                                    
                                    # Fit a down facing parabola to the above points (x1, y1), (x2, y2), (x3, y3)
                                    # y = a*x*x + b*x + c
                                    # y1 = a*x1*x1 + b*x1 + c
                                    # y1 = a*x2*x2 + b*x2 + c
                                    # y1 = a*x3*x3 + b*x3 + c
                                    # Y = X * Transpose([a, b, c]) (= L, say)
                                    # L = inverse(X) * Y
                                    
                                    X = np.array([
                                        [x1*x1, x1, 1],
                                        [x2*x2, x2, 1],
                                        [x3*x3, x3, 1]
                                    ])
                                    
                                    Y = np.array([
                                        [y1],
                                        [y2],
                                        [y3]
                                    ])
                                    
                                    L = np.dot(np.invert(X), Y)
                                    
                                    # So, now we have a,b,c for our parabola equation,let's find vertex
                                    x0 = -L[1] / (2 * L[0])
                                    if math.fabs(x0) > 2 * self.bins: x0 = x2
                                    while(x0 < 0): x0 += self.bins
                                    while(x0 > self.bins): x0 -= self.bins
                                    
                                    x0_normalized = x0 * (2 * self.pi / self.bins)
                                    x0_normalized -= self.pi
                                    
                                    o.append(x0_normalized)
                                    m.append(orientation_hist[k])
                            
                            self.keypoints.append(Keypoint(
                                x * scale / 2.0, y * scale / 2.0, m, o, i * self.intervals + j - 1
                            ))

    def generate_features(self):
        magnitudes_interpolated = [
            [
                None for interval in range(self.intervals)
            ] for octave in range(self.octaves)
        ]
        orientations_interpolated = [
            [
                None for interval in range(self.intervals)
            ] for octave in range(self.octaves)
        ]
        
        for i in range(self.octaves):
            for j in range(1, self.intervals + 1):
                temp = np.float32(cv2.pyrUp(self.gaussian[i][j]))
                
                magnitudes_interpolated[i][j - 1] = np.float32(
                    np.zeros_like(
                        cv2.resize(
                            self.gaussian[i][j],
                            (self.gaussian[i][j].shape[1] + 1, self.gaussian[i][j].shape[0] + 1)
                        )
                    )
                )
                orientations_interpolated[i][j - 1] = np.float32(
                    np.zeros_like(
                        cv2.resize(
                            self.gaussian[i][j],
                            (self.gaussian[i][j].shape[1] + 1, self.gaussian[i][j].shape[0] + 1)
                        )
                    )
                )
                
                ii = 1.5
                while(ii < self.gaussian[i][j].shape[0] - 1.5):
                    jj = 1.5
                    
                    while(jj < self.gaussian[i][j].shape[1] - 1.5):
                        ii1 = int(ii + 1.5)
                        ii2 = int(ii + 0.5)
                        ii3 = int(ii)
                        ii4 = int(ii - 0.5)
                        ii5 = int(ii - 1.5)
                        
                        jj1 = int(jj + 1.5)
                        jj2 = int(jj + 0.5)
                        jj3 = int(jj)
                        jj4 = int(jj - 0.5)
                        jj5 = int(jj - 1.5)
                        
                        dx = (
                            (self.gaussian[i][j][ii1, jj3] + self.gaussian[i][j][ii2, jj3]) / 2.0 -
                            (self.gaussian[i][j][ii5, jj3] + self.gaussian[i][j][ii4, jj3]) / 2.0
                        )
                        dy = (
                            (self.gaussian[i][j][ii3, jj1] + self.gaussian[i][j][ii3, jj2]) / 2.0 -
                            (self.gaussian[i][j][ii3, jj5] + self.gaussian[i][j][ii3, jj4]) / 2.0
                        )
                        
                        x_ = int(ii + 1)
                        y_ = int(jj + 1)
                        
                        magnitudes_interpolated[i][j - 1][x_, y_] = math.sqrt(dx ** 2 + dy ** 2)
                        if math.atan2(dy, dx) == self.pi: orientations_interpolated[i][j - 1][x_, y_] = -self.pi
                        else: orientations_interpolated[i][j - 1][x_, y_] = math.atan2(dy, dx)
                        
                        jj += 1
                    ii += 1
                
                for ii in range(self.gaussian[i][j].shape[0] + 1):
                    magnitudes_interpolated[i][j - 1][ii, 0] = 0
                    magnitudes_interpolated[i][j - 1][ii, self.gaussian[i][j].shape[1] - 1] = 0
                    orientations_interpolated[i][j - 1][ii, 0] = 0
                    orientations_interpolated[i][j - 1][ii, self.gaussian[i][j].shape[1] - 1] = 0
                
                for jj in range(self.gaussian[i][j].shape[1] + 1):
                    magnitudes_interpolated[i][j - 1][0, jj] = 0
                    magnitudes_interpolated[i][j - 1][self.gaussian[i][j].shape[0] - 1, jj] = 0
                    orientations_interpolated[i][j - 1][0, jj] = 0
                    orientations_interpolated[i][j - 1][self.gaussian[i][j].shape[0] - 1, jj] = 0
        
        G = self.interpolated_gaussian(self.feature_win_dim, 0.5 * self.feature_win_dim)
        
        buggy_keypoints = []

        for keypoint in self.keypoints:
            scale = keypoint.scale
            x = keypoint.x
            y = keypoint.y
            
            ii = int(x * 2) / int(2.0 ** (scale / self.intervals))
            jj = int(y * 2) / int(2.0 ** (scale / self.intervals))
            
            orientations = keypoint.orientation
            magnitudes = keypoint.magnitude
            
            main_orientation = orientations[0]
            main_magnitude = magnitudes[0]
            
            for i in range(len(magnitudes)):
                if magnitudes[i] > main_magnitude:
                    main_orientation = orientations[i]
                    main_magnitude = magnitudes[i]
            
            half_kernel_size = self.feature_win_dim / 2
            weights = np.float32(np.zeros((self.feature_win_dim, self.feature_win_dim)))
            
            for i in range(self.feature_win_dim):
                for j in range(self.feature_win_dim):
                    if ii + i + 1 < half_kernel_size:
                        weights[i, j] = 0
                    elif ii + i + 1 > half_kernel_size + self.gaussian[scale / self.intervals][0].shape[1]:
                        weights[i, j] = 0
                    elif jj + j + 1 < half_kernel_size:
                        weights[i, j] = 0
                    elif jj + j + 1 > half_kernel_size + self.gaussian[scale / self.intervals][0].shape[0]:
                        weights[i, j] = 0
                    else:
                        val = magnitudes_interpolated[scale / self.intervals][scale % self.intervals]
                        val = val[ii + i + 1 - half_kernel_size, jj + j + 1 - half_kernel_size]
                        
                        weights[i, j] = (G[i, j] * val)
            
            # 16 4x4 blocks
            feature_vector = [0.0 for s in range(self.feature_vector_size)]
            for i in range(self.feature_win_dim/4):
                for j in range(self.feature_win_dim/4):
                    hist = [0.0 for b in range(self.descriptor_bins)]
                    
                    start_i = int(ii - half_kernel_size) + 1 + int(half_kernel_size / 2 * i)
                    start_j = int(jj - half_kernel_size) + 1 + int(half_kernel_size / 2 * j)
                    
                    limit_i = int(ii) + int(half_kernel_size / 2) * (i - 1)
                    limit_j = int(jj) + int(half_kernel_size / 2) * (j - 1)
                    
                    for iii in range(start_i, limit_i + 1):
                        for jjj in range(start_j, limit_j + 1):
                            if iii < 0 or iii >= self.gaussian[scale / self.intervals][0].shape[1]: continue
                            if jjj < 0 or jjj >= self.gaussian[scale / self.intervals][0].shape[0]: continue
                            
                            # Rotation invariance
                            sampled = orientations_interpolated[scale / self.intervals][scale % self.intervals]
                            
                            sampled_orientation = sampled[iii, jjj] - main_orientation
                            while sampled_orientation < 0: sampled_orientation += (2 * self.pi)
                            while sampled_orientation > 2 * self.pi: sampled_orientation -= (2 * self.pi)
                            
                            degrees = sampled_orientation * 180 / self.pi
                            bin = degrees * self.descriptor_bins / 360.0
                            
                            w = weights[iii + half_kernel_size - ii - 1, jjj + half_kernel_size - jj - 1]
                            hist[int(bin)] += (1 - math.fabs(bin - int(bin) - 0.5)) * w
                    
                    for k in range(self.descriptor_bins):
                        feature_vector[(i * self.feature_win_dim / 4 + j) * self.descriptor_bins + k] += hist[k]
            
            # Illumination invariance
            try:
                norm = 0
                for i in range(self.feature_vector_size):
                    norm += (feature_vector[i] ** 2)
                norm = math.sqrt(norm)

                for i in range(self.feature_vector_size):
                    feature_vector[i] /= norm
                    if feature_vector[i] > self.feature_vector_threshold:
                        feature_vector[i] = self.feature_vector_threshold

                norm = 0
                for i in range(self.feature_vector_size):
                    norm += feature_vector[i] ** 2
                norm = math.sqrt(norm)

                for i in range(self.feature_vector_size):
                    feature_vector[i] /= norm

                self.descriptors.append(Descriptor(x, y, feature_vector))
            except:
                buggy_keypoints.append(keypoint)
                self.num_keypoints -= 1
        
        for bug in buggy_keypoints:
            self.keypoints.remove(bug)
                
    
    def draw_keypoints(self):
        img = self.image.copy()
        
        for kp in self.keypoints:
            r = int(random.random() * 500)
            while r > 255: r -= 50
            g = int(random.random() * 500)
            while g > 255: g -= 50
            b = int(random.random() * 500)
            while b > 255: b -= 50
            
            color = (b, g, r)
            
            cv2.line(
                img,
                (int(kp.x), int(kp.y)),
                (int(kp.x + 10 * math.cos(kp.orientation[0])), int(kp.y + 10 * math.sin(kp.orientation[0]))),
                color,
                2
            )
            cv2.circle(
                img,
                (int(kp.x), int(kp.y)),
                10,
                color,
                2
            )
        
        return img
                
    def interpolated_gaussian(self, size, sigma):
        half_size = size / 2 - 0.5
        sog = 0
        ret = np.float32(np.zeros((size, size), dtype=np.float32))
        
        for i in range(size):
            for j in range(size):
                x, y = i - half_size, j - half_size
                temp = 1.0 / (2 * self.pi * (sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2.0 * (sigma ** 2)))
                ret[i, j] = temp
                sog += temp
        
        for i in range(size):
            for j in range(size):
                ret[i, j] *= (1.0 / sog)
        
        return ret


image = cv2.imread("../resources/messi.jpg")

sift = SIFT(image, 4, 2)

sift.build_scale_space()
sift.detect_extrema()
sift.assign_orientation()
sift.generate_features()

out = sift.draw_keypoints()

fig = plt.figure()
fig.set_size_inches(10, 10)
fig.add_subplot(1,1,1)
plt.imshow(out)
plt.show()





import cv2
import numpy as np
import matplotlib.pyplot as plt


# ## Features Tracking and Detection
# Features are specific patterns in an image which are unique, can be easily tracked and compared. Corners are usually the best features as they are the regions with large variation in intensity in all directions. They are intutively junctions of contours.
# 
# Useful in some vision tasks like stereo and motion estimation, where it's required to find corresponding features across two or more views.
# 
# ### Harris Corner Detection
# Difference in intensity for a displacement of 'x' in x-direction and 'y' in y-direction, in all directions, is given by
# $$E(x, y) = \sum_{u}\sum_{v} w(u, v) * [I(u + x, v + y) - I(u, v)]^2$$
# - w(u, v) is window function (rectangular or gaussian window which gives weights to pixels underneath, its used for the convolution).
# - I(u, v) is intensity of pixel located at coordinate (x, y)
# 
# E(u, v) is maximized to detect corners.
# $$\text{Let }I_{x}\text{ and }I_{y}\text{ be the partial derivatives of I, I(u + x, v + y) can be approximated using taylor series}$$
# $$I(u + x, v + y) \approx I(u, v) + I_{x}(u, v)*x + I_{y}(u, v)*y$$
# $$E(x, y) \approx \sum_{u}\sum_{v} w(u, v) * [I_{x}(u, v)*x + I_{y}(u, v)*y]^2$$
# Representing the above equation in matrix form
# $$E(x, y) \approx \begin{bmatrix}x & y\end{bmatrix} * M\begin{bmatrix}x\\y\end{bmatrix}$$
# $$M = \sum_{u, v} w(u, v) * \begin{bmatrix}I_{x}^2 & I_{x}I_{y}\\I_{x}I_{y} & I_{y}^2\end{bmatrix}$$
# $$\text{M will have 2 eigen values }\lambda_{1}\text{ and }\lambda_{2}$$
# 

edge = cv2.imread("../resources/harris_1.png", cv2.IMREAD_GRAYSCALE)
edge_x = cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=5)
edge_y = cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=5)

flat = cv2.imread("../resources/harris_2.png", cv2.IMREAD_GRAYSCALE)
flat_x = cv2.Sobel(flat, cv2.CV_64F, 1, 0, ksize=5)
flat_y = cv2.Sobel(flat, cv2.CV_64F, 0, 1, ksize=5)

corner = cv2.imread("../resources/harris_3.png", cv2.IMREAD_GRAYSCALE)
corner_x = cv2.Sobel(corner, cv2.CV_64F, 1, 0, ksize=5)
corner_y = cv2.Sobel(corner, cv2.CV_64F, 0, 1, ksize=5)

plt.subplot(3, 3, 1)
plt.imshow(edge, cmap='gray')
plt.subplot(3, 3, 2)
plt.imshow(edge_x, cmap='gray')
plt.subplot(3, 3, 3)
plt.imshow(edge_y, cmap='gray')

plt.subplot(3, 3, 4)
plt.imshow(flat, cmap='gray')
plt.subplot(3, 3, 5)
plt.imshow(flat_x, cmap='gray')
plt.subplot(3, 3, 6)
plt.imshow(flat_y, cmap='gray')

plt.subplot(3, 3, 7)
plt.imshow(corner, cmap='gray')
plt.subplot(3, 3, 8)
plt.imshow(corner_x, cmap='gray')
plt.subplot(3, 3, 9)
plt.imshow(corner_y, cmap='gray')

plt.show()


fig = plt.figure()
fig.set_size_inches(18, 5)
fig.add_subplot(1, 3, 1)
plt.scatter(edge_x.flatten(), edge_y.flatten())
fig.add_subplot(1, 3, 2)
plt.scatter(flat_x.flatten(), flat_y.flatten())
fig.add_subplot(1, 3, 3)
plt.scatter(corner_x.flatten(), corner_y.flatten())
plt.show()


# Fitting a confidence ellipse around the scatter plot for the above cases, we notice
# 1. For edge region,
# $$\lambda_{1}>>\lambda_{2}\text{ or vice-versa}$$
# 2. For flat region,
# $$\lambda_{1} \approx \lambda_{2}$$
# 3. For corner region,
# $$\lambda_{1} \approx \lambda_{2}\text{ and the eigen values are comparatively larger than those for flat region}$$
# 
# Score equation to determine if a window contains a corner: <b>R = det(M) - k(trace(M))<sup>2</sup></b>
# $$det(M) = \lambda_{1}\lambda_{2}$$
# $$trace(M) = \lambda_{1} + \lambda_{2}$$
# - R is small => flat region
# - R < 0 => edge region
# - R is large => corner region
# <img src="../resources/harris.jpg">
# Function: <b>cv2.cornerHarris(image, window_block_size, sobel_block_size, k)</b>
# 

image = cv2.imread("../resources/corner_test.jpg")
copy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fig = plt.figure()
fig.set_size_inches(10, 5)
fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

corner_regions = cv2.cornerHarris(gray, 3, 3, 0.04)
thresholded_region = corner_regions > 0.01 * corner_regions.max()
# print "Corners: ", image[thresholded_region]
image[thresholded_region] = [0, 255, 0]

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()


# ### Refined Corners
# Function: <b>cv2.cornerSubPixel(image, corners, winSize, zeroZone, criteria)</b>
# - winSize: Half of side length of search window, (3, 3) => (3*2+11 3*2+1) => (7, 7)
# - zeroZone: Half of size of dead region in middle of searc window
# - criteria: Termination of the iterative process of corner refinement. Stopping parameters => maxCount and epsilon.
# 
# With a corner detection algorithm like the Harris corner detector, we end up with a corner like (56, 120). But, sometimes we want a more precise corner like (56.768, 120.1432).
# 

# Refined Corners
# Use of cv2.cornerSubPixel

corner_regions = cv2.dilate(corner_regions, None)
refined_region = cv2.threshold(corner_regions, 0.001 * corner_regions.max(), 255, cv2.THRESH_BINARY)[1]
refined_region = np.uint8(refined_region)

ret, labels, stats, centroids = cv2.connectedComponentsWithStats(refined_region)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000000, 0.0001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (3, 3), (-1, -1), criteria)

res = np.int0(np.hstack((centroids, corners)))
copy[res[:, 1], res[:, 0]] = [255, 0, 0]
copy[res[:, 3], res[:, 2]] = [0, 255, 0]

plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
plt.show()


# ### Shi-Tomasi Corner Detection
# The Shi-Tomasi corner detector is based entirely on the Harris corner detector. However, one slight variation in "selection criteria" made this detector much better than the original. It works quite well where even the Harris corner detector fails.
# $$Score (R) = min(\lambda_{1}, \lambda_{2})$$
# <img src="../resources/shi-tomasi.jpg">
# 
# - <font color="#00FF00">Corner Region</font> $$\text{Both }\lambda_{1}\text{ and }\lambda_{2}\text{ are greater than threshold value}$$
# - <font color="#3311FF">Edge <font color="#808080">Region</font></font> $$\text{Either }\lambda_{1}\text{ or }\lambda_{2}\text{ is smaller than threshold value}$$
# - <font color="#FF69B4">Flat Region</font>$$\text{Both }\lambda_{1}\text{ and }\lambda_{2}\text{ are less than threshold value}$$
# 

image = cv2.imread("../resources/corner_test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.goodFeaturesToTrack(image, number of corners, quality, min euclidean distance)
tomasi_corners = np.int0(cv2.goodFeaturesToTrack(gray, 20, 0.05, 8))
for corner in tomasi_corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 4, (0, 255, 0), 2)

fig = plt.figure()
fig.set_size_inches(5, 10)
fig.add_subplot(1, 1, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()





import cv2
import numpy as np
from matplotlib import pyplot as plt


# ## Smoothing / Blurring
# Aim is to supress noise in image. It is done by convoluting a selected kernel(filter) with the image.
# 
# ### Averaging
# Here, kernel is a normalized box. It takes average of all pixels under kernel area and assigns this mean value to the central pixel.
# 
# Functions: <b>cv2.blur(image, kernel_size)</b>, <b>cv2.boxFilter(image, depth, kernel_size)</b>
# 

image = cv2.imread('../resources/lena.jpg')

blurred1 = cv2.blur(image, (5, 5))
blurred2 = cv2.boxFilter(image, -1, (5, 5))  # -1 means same depth as source image

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(blurred1, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(blurred2, cv2.COLOR_BGR2RGB))

plt.show()


# ### Gaussian Filtering
# It uses a gaussian kernel, and is highly efficient for gaussian noise (as the name suggests). It is most commonly used in edge detection.
# 
# <img src="../resources/gauss.png">
# 
# Function: <b>cv2.GaussianBlur(image, kernel_size, sigmaX[, sigmaY])</b>
# 

gauss_blur = cv2.GaussianBlur(image, (9, 9), 3)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2RGB))

plt.show()


# ### Median Filtering
# As the name suggests, it computes median of all the pixels under the kernel area and assigns this value to the central pixel. It is highly effective in remove salt and pepper noise. [* Each altered pixel value is some pixel from image itself, thus reducing noise very effectively]
# 
# Function: <b>cv2.medianBlur(image, kernel_size)</b>
# 

extra_noise = cv2.imread('../resources/noise.jpg')

median_blur = cv2.medianBlur(extra_noise, 5)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(extra_noise, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))

plt.show()


# ### Bilateral Filtering
# Slow but effective in noise suppression without blurring edges. As the name suggests, it uses two filters (both gaussian), one in the space domain, and other as a pixel-intensity-difference function. Thus, in addition to normal gaussian filtering, it ensures that only those pixels with similar intensity to central pixel are included in the computation, and hence preserving edges from getting blurred.
# 
# Function: <b>cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)</b> (keep sigma values between 10 and 150)
# - d - diameter of pixel neighbourhood used during filtering
# - sigmaColor - Filter sigma in color space
# - sigmaSpace - Filter sigma in coordinate space
# 

bi_blur = cv2.bilateralFilter(image, 9, 120, 100)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(bi_blur, cv2.COLOR_BGR2RGB))

plt.show()


# ## Morphological Transformations
# Binary images produced after thresholding, are distored by noise and texture. Morphology transformations are some simple operations based on image shape. Morphological operations rely on relative ordering of pixel values rather than their numerical values, and therefore are suitable for processing of binary images.
# 
# In morphological transformation, we traverse an image with a structuring element (a small binary image, ie, a small matrix of pixels, each with value 0 or 1), and test various relations between the element and regions of image.
# 
# We'll be discussing following types of Morphological Operators:
# 1. Erosion
# 2. Dilation
# 3. Opening
# 4. Closing
# 5. Morphological Gradient
# 

# let's have a look at structuring elements first
# 3 major structure elements - rectangle, ellipse, cross

# Rectangular Kernel
print "rectangle"
print cv2.getStructuringElement(cv2.MORPH_RECT, (8, 10))
print
# Elliptical Kernel
print "ellipse"
print cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 10))
print
# Crossed Kernel
print "cross"
print cv2.getStructuringElement(cv2.MORPH_CROSS, (8, 10))
print


# ### Erosion
# Erodes away the boundaries of foreground object. As kernel is slided over image, central pixel is assigned 1 only if all pixels under kernel area are 1, otherwise 0. This dicreases the thickness of foreground object.
# 

letter = cv2.imread('../resources/letter1.png', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded = cv2.erode(letter, kernel, iterations=1)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(letter, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(eroded, cmap='gray')

plt.show()


# ### Dilation
# It's opposite of erosion. Here, even if one pixel under kernel area is 1, the central pixel is assigned 1 as well. Hence, our foreground object is dilated.
# 

dilated = cv2.dilate(letter, kernel, iterations=1)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(letter, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(dilated, cmap='gray')

plt.show()


# ### Opening
# Erosion followed by dilation. Erosion removes white noises, and shrinks our foreground object, while dilation dilates our eroded object.
# 

opening = cv2.morphologyEx(letter, cv2.MORPH_OPEN, kernel)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(letter, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(opening, cmap='gray')

plt.show()


# ### Closing
# Dilation followed by Erosion. Used to fill holes in our foreground object.
# 

letter = cv2.imread('../resources/letter2.png', cv2.IMREAD_GRAYSCALE)

closing = cv2.morphologyEx(letter, cv2.MORPH_CLOSE, kernel)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(letter, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(closing, cmap='gray')

plt.show()


# ### Morphological Gradient
# Difference between dilation and erosion. Gives us outline of the foreground object
# 

letter = cv2.imread('../resources/letter1.png', cv2.IMREAD_GRAYSCALE)

outline = cv2.morphologyEx(letter, cv2.MORPH_GRADIENT, kernel)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(letter, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(outline, cmap='gray')

plt.show()


import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('../resources/messi.jpg')


# ## Thresholding
# A simple segmentation method used to separate out regions of an image corresponding to objects we want to analyze.
# ### Simple Thresholding
# If a pxel value is greater than a threshold, it is assigned one value (maybe white), else  another value (maybe black).
# 
# Function: <b>`cv2.threshold(image, threshold_value, max_value, thresholding_type)`</b>
# 
# <u>Types of thresholding</u>:
# 1. <b>cv2.THRESH_BINARY</b> - If pixel value above threshold, it is assigned max_value, else 0. (default flag)
# 2. <b>cv2.THRESH_BINARY_INV</b> - Opposite of cv2.THRESH_BINARY. Pixel value above threshold is assigned 0 and vice-versa.
# 3. <b>cv2.THRESH_TRUNC</b> - Pixel values above threshold are truncated to threshold value.
# 4. <b>cv2.THRESH_TOZERO</b> - If pixel value above threshold, remains unchanges. Otherwise, set to 0.
# 5. <b>cv2.THRESH_TOZERO_INV</b> - Opposite of cv2.THRESH_TOZERO. If pixel value is above threshold, will be set to 0.
# 

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
thresh2 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]
thresh3 = cv2.threshold(gray, 80, 255, cv2.THRESH_TRUNC)[1]
thresh4 = cv2.threshold(gray, 80, 255, cv2.THRESH_TOZERO)[1]
thresh5 = cv2.threshold(gray, 80, 255, cv2.THRESH_TOZERO_INV)[1]

titles = ['GRAY', 'TRUNC', 'BINARY', 'BINARY_INV', 'TOZERO', 'TOZERO_INV']
plots = [gray, thresh3, thresh1, thresh2, thresh4, thresh5]

fig = plt.figure()
fig.set_size_inches(30, 30)

for i in range(6):
    fig.add_subplot(3, 2, i+1)
    plt.imshow(plots[i], cmap='gray')
    plt.title(titles[i])

plt.show()


# ### Adaptive Thresholding
# Rather than applying same threshold value across the image, threshold value at each pixel location is calculated depending on the neighbouring pixel intensities.
# 
# Function: <b>`cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)`</b>
# 
# <u>Adaptive Methods</u>:
# 1. <b>cv2.ADAPTIVE_THRESH_MEAN_C</b> - threshold is mean of neighbourhood area
# 2. <b>cv2.ADAPTIVE_THRESH_GAUSSIAN_C</b> - threshold is weighted sum of neighbourhood area
# 
# <b>Block Size</b> - Size of neighbourhood area
# 
# <b>C</b> - A constant subtracted from threshold calculated
# 

mean_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)
gaussian_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 0)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(mean_thresh, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(gaussian_thresh, cmap='gray')

plt.show()


# ### Otsu's Thresholding
# Automatically calculates threshold value from a bimodal image. (Accurate only for bimodal images)
# 

thresh_selected, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print "Calculated threshold: %f" % (thresh_selected)

plt.imshow(otsu_thresh, cmap='gray')
plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt


# # Pyramid
# It's a type of multi-scale signal representation, in which a signal or an image is subjected to repeated smoothing and subsampling.
# 
# Subsampling is done to reduce amount of data in the spacial domain, that needs to be transmitted, stored or processed. As human visual system is less sensitive for color information than the intensity information (brightness), in a Y-Cr-Cb colorspace, Cr and Cb components (colors) are filtered and sub-sampled at fraction of Y (luma) component.
# 
# In short, set of images with different resolutions(different frequency-band images), stacked wtih biggest image at bottom and smallest at top, are called Image Pyramids.
# <img src="../resources/pyramid.png">
# Two main types of Image Pyramids:
# 1. Lowpass - Made by smoothing the image with an appropriate smoothing filter and then subsampling the smoothed image, usually by a factor of 2 along each coordinate direction.
# 2. Bandpass - Made by forming the difference between images at adjacent levels in the pyramid and performing some kind of image interpolation between adjacent levels of resolution, to enable computation of pixelwise differences.
# 
# 
# - Guassian Pyramid - It's a lowpass pyramid. As we move from lower level (high resolution) to higher level (low resolution), images are weighted down using a Gaussian Blur and scaled down. Each pixel contains a local average that corresponds to a pixel neighborhood on a lower level of the pyramid. At each level, image size is reduced by a factor of 0.5. (It's better to perform convolution with one signle image being resized again and again rather than performing convolutions of image wtih various kernels to get different level outputs)
# - Laplacian Pyramid - It's a bandpass pyramid. Each layer is made from difference between that level in Gaussian Pyramid and expanded version of it's upper level in Gaussian Pyramid.
# 
# 
# Uses of Pyramids
# 1. Laplacian Pyramid prodcues edge images.
# 2. Laplacian Pyramid is used in image compression.
# 3. Image blending.
# 
# Pyramids are great for image representation as they have both spatial-frequency domain and are easy to compute.
# 

image = cv2.imread("../resources/messi.jpg")

# Gaussian Pyramid
lower_reso1 = cv2.pyrDown(image)
original = cv2.pyrUp(lower_reso1)  # will be blurred due to image smoothing (some information is lost)

fig = plt.figure()
fig.set_size_inches(18, 10)
fig.add_subplot(1, 2, 1)
plt.imshow(image)
fig.add_subplot(1, 2, 2)
plt.imshow(original)
plt.show()


# Laplacian Pyramid
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
down1 = cv2.pyrDown(gray)
down2 = cv2.pyrDown(down1)

down2 = cv2.resize(down2, (down1.shape[1], down1.shape[0]), interpolation=cv2.INTER_LINEAR)
laplacian2 = cv2.subtract(down1, down2)

down1 = cv2.resize(down1, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
laplacian1 = cv2.subtract(gray, down1)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(laplacian1, cmap='gray')
fig.add_subplot(1, 2, 2)
plt.imshow(laplacian2, cmap='gray')

plt.show()


# Image Blending
orange = cv2.imread("../resources/orange.jpg")
apple = cv2.imread("../resources/apple.jpg")

orange_copy = orange.copy()
apple_copy = apple.copy()
gauss_orange = [orange_copy]
gauss_apple = [apple_copy]
for i in range(5):
    orange_copy = cv2.pyrDown(orange_copy)
    gauss_orange.append(orange_copy)
    
    apple_copy = cv2.pyrDown(apple_copy)
    gauss_apple.append(apple_copy)

laplacian_orange = [gauss_orange[4]]
laplacian_apple = [gauss_apple[4]]
for i in range(4, 0, -1):
    higher_orange = cv2.pyrUp(gauss_orange[i], dstsize=(gauss_orange[i-1].shape[1], gauss_orange[i-1].shape[0]))
    diff_orange = cv2.subtract(gauss_orange[i-1], higher_orange)
    laplacian_orange.append(diff_orange)
    
    higher_apple = cv2.pyrUp(gauss_apple[i], dstsize=(gauss_apple[i-1].shape[1], gauss_apple[i-1].shape[0]))
    diff_apple = cv2.subtract(gauss_apple[i-1], higher_apple)
    laplacian_apple.append(diff_apple)

blends = []
for lap_orange, lap_apple in zip(laplacian_orange, laplacian_apple):
    rows, cols, depth = lap_orange.shape
    blend = np.hstack((lap_orange[:, 0:cols/2], lap_apple[:, cols/2:]))
    blends.append(blend)

reconstructed = blends[0]
for i in range(1, 5):
    reconstructed = cv2.pyrUp(reconstructed, dstsize=(blends[i].shape[1], blends[i].shape[0]))
    reconstructed = cv2.add(reconstructed, blends[i])

plt.imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
plt.show()


# Cartoon Effect
# better to downsample first, reduce color information and then upsample
# down-sample the image
image = cv2.imread("../resources/scene.jpg")
color = image.copy()
for i in range(2):
    color = cv2.pyrDown(color)

# better to apply small filter repeatedly than apply a slow big filter
for i in range(7):
    color = cv2.bilateralFilter(color, 9, 9, 7)

# up-sample the image
for i in range(2):
    color = cv2.pyrUp(color)

# reduce noise
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 7)

edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

color = cv2.resize(color, (edges.shape[1], edges.shape[0]))
cartoon = cv2.bitwise_and(color, edges)

fig = plt.figure()
fig.set_size_inches(10, 8)

fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))

plt.show()





