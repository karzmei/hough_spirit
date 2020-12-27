# Detect circles on a gray-scale image (first of a specific radius).
# Using Hough transform idea: vote for points that are likely to be a center of a circle (of a given radius), 
# producing a heat-map of the given image, with more-likely-to-be-center getting higher votes.

import numpy as np
from PIL import Image  # for working with images
from scipy import ndimage	# manipulation on images using np.arrays and scipy

R = 10
GRAY = 256	# we'd want to do computations modulu GRAY

SMALL = np.array([[0,0,0,0,0], \
				 [0,0,1,0,0], \
				 [0,1,1,1,0], \
				 [0,0,1,0,0], \
				 [0,0,0,0,0]])


def naive_gradient_size_per_point(pix):
	""" Computes size of the gradient at every pixel, except last row and last column.
	The gradient is computed using differences with +1 pixels in rows and cols directions."""
	grad_sz = np.zeros_like(pix, dtype = float)

	d_rows = - pix[:-1, :-1] + pix[1:,  :-1]
	d_cols = - pix[:-1, :-1] + pix[ :-1, 1:]

	grad_sz[:-1, :-1] = np.sqrt(d_rows**2 + d_cols**2)
	return grad_sz

def rescale_arr(arr, end = GRAY):
	""" Scales (positive) arr values to 0-end integer values."""
	arr = np.round( (arr - np.min(arr)) * (end -1) / (np.max(arr) - np.min(arr)) )
	return arr

def bw_by_threshold(arr, threshold, low_val = 0, high_val = GRAY - 1):
	""" Sets low-values elements of arr to be low_val (deault = 0), 
	and high-valued to be val, the default being the highest gray-scale "color"."""
	low_idx = arr < threshold
	high_idx = arr >= threshold
	arr[low_idx] = low_val
	arr[high_idx] = high_val
	return arr

def gradients_map():
	pass

def produce_heat_map():
	pass


if __name__ == '__main__':
	# load image:
	img = Image.open('pics/philippe-leone-circles_hard.jpg')	#wiki_circle_rainbow_effect.jpg')
	width, height = img.size
	#turn into gray-scale:
	img = img.convert('L')
	#img.show()

	# convert to numpy array
	pix = np.array(img)
	height, width = pix.shape	# same sizes as before


	grads_sz = gradient_size_per_point(pix)
	print(np.max(grads_sz), np.min(grads_sz))
	grads_sz = rescale_arr(grads_sz)

	test_img = Image.fromarray(grads_sz)
	img.show()
	test_img.show()
	# example of backwards conversion:
	#result_img = Image.fromarray(pix)
	# result_img.show

