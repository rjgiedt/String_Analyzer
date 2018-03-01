#!/usr/bin/env python

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import PIL
import numpy as np
import matplotlib.pylab as plt
from PIL import ImageFilter
from PIL import ImageEnhance
from skimage import morphology
from skimage import data, util
from skimage.measure import label
from skimage.measure import regionprops
from scipy import ndimage

# Define Image Locations/ Names 
Image_Loc = '/Users/jimmy/Desktop/'

def bradley_roth_numpy(image, s=None, t=None):

    # Convert image to numpy array
    img = np.array(image).astype(np.float)

    # Default window size is round(cols/8)
    if s is None:
        s = np.round(img.shape[1]/8)

    # Default threshold is 15% of the total
    # area in the window
    if t is None:
        t = 15.0

    # Compute integral image
    intImage = np.cumsum(np.cumsum(img, axis=1), axis=0)

    # Define grid of points
    (rows,cols) = img.shape[:2]
    (X,Y) = np.meshgrid(np.arange(cols), np.arange(rows))

    # Make into 1D grid of coordinates for easier access
    X = X.ravel()
    Y = Y.ravel()

    # Ensure s is even so that we are able to index into the image
    # properly
    s = s + np.mod(s,2)

    # Access the four corners of each neighbourhood
    x1 = X - s/2
    x2 = X + s/2
    y1 = Y - s/2
    y2 = Y + s/2

    # Ensure no coordinates are out of bounds
    x1[x1 < 0] = 0
    x2[x2 >= cols] = cols-1
    y1[y1 < 0] = 0
    y2[y2 >= rows] = rows-1

    # Count how many pixels are in each neighbourhood
    count = (x2 - x1) * (y2 - y1)

    # Compute the row and column coordinates to access
    # each corner of the neighbourhood for the integral image
    f1_x = x2
    f1_y = y2
    f2_x = x2
    f2_y = y1 - 1
    f2_y[f2_y < 0] = 0
    f3_x = x1-1
    f3_x[f3_x < 0] = 0
    f3_y = y2
    f4_x = f3_x
    f4_y = f2_y

    # Compute areas of each window
    sums = intImage[f1_y, f1_x] - intImage[f2_y, f2_x] - intImage[f3_y, f3_x] + intImage[f4_y, f4_x]

    # Compute thresholded image and reshape into a 2D grid
    out = np.ones(rows*cols, dtype=np.bool)
    out[img.ravel()*count <= sums*(100.0 - t)/100.0] = False

    # Also convert back to uint8
    out = 255*np.reshape(out, (rows, cols)).astype(np.uint8)

    # Return PIL image back to user
    return Image.fromarray(out)

# Load Image (from 16 bit tiff)
initial_image = plt.imread(Image_Loc)
im2 = (initial_image/8).astype('uint8')
im = Image.fromarray(im2)

###########################################################################
# Load Image from 8 bit jpeg
#im = Image.open('/Users/jimmy/Desktop/Combined_Image_5.jpg')

# Image preprocessing
# Unsharp Mask
fil = PIL.ImageFilter.UnsharpMask(5, 150, 3)
filtered = im.filter(fil)

medfil = PIL.ImageFilter.MedianFilter(5)
filtered2 = filtered.filter(medfil)

edge_enhance = PIL.ImageFilter.EDGE_ENHANCE_MORE()
filtered3 = filtered2.filter(edge_enhance)

fil2 = PIL.ImageFilter.UnsharpMask(5, 150, 3)
filtered4 = filtered3.filter(fil)

filtered5 = np.array(filtered4)
cleaned = morphology.remove_small_objects(filtered5, min_size=5)
cleaned2 = PIL.Image.fromarray(cleaned)

# Thresholding and Filtering
bradley = bradley_roth_numpy(cleaned2, t = 5, s = 300)
bradley2 = np.array(bradley, bool)
size_filter = morphology.remove_small_objects(bradley2, min_size=25)
#open_close = ndimage.binary_closing(open_close, structure=np.ones((5,5))).astype(np.int)
open_close = ndimage.binary_opening(size_filter, structure=np.ones((3,3))).astype(np.int)
   
#Open-Close
final = np.array(open_close*255)
final = np.uint8(final)
final2 = Image.fromarray(final)
#final2.show()
    
# Find Image Properties
im = np.array(final)
label_img = label(im)
props = regionprops(label_img)

Major_axes = []
Areas = []
Form_Factors = []
Perimeters = []

for structure in range(len(props)):
    if props[structure]['major_axis_length'] < 1:
        Major_axes.append(1)
    else:
        Major_axes.append(props[structure]['major_axis_length'])
    if props[structure]['perimeter'] < 1:
        Perimeters.append(1)
    else:
        Perimeters.append(props[structure]['perimeter'])
    
for structure in range(len(props)):    
    Areas.append(props[structure]['area'])
    Form_Factor = np.square(Perimeters[structure])/(4*np.pi*props[structure]['area'])
    Form_Factors.append(Form_Factor)
    Perimeters.append(props[structure]['perimeter'])
 
#Calculate Percentiles for structures found                     
Major_axes_99 = np.percentile(Major_axes, 99.5)
Areas_99 = np.percentile(Areas, 99.5)
Form_Factors_99 = np.percentile(Form_Factors, 99.5)
Perimeters_99 = np.percentile(Perimeters, 99.5)
                      
FF_MA = Major_axes_99*Form_Factors_99

Final_FF_MA = []
if FF_MA > 2500:
	Final_FF_MA = 2500
else:
	Final_FF_MA = FF_MA

print Form_Factors_99
print Major_axes_99
print Final_FF_MA