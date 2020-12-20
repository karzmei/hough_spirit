# Detect circles on a gray-scale image (first of a specific radius).
# Using Hough transform idea: vote for points that are likely to be a center of a circle (of a given radius), 
# producing a heat-map of the given image, with more-likely-to-be-center getting higher votes.

import numpy as np
from PIL import Image  # for working with images

R = 10


def gradient_at_point():
	pass

def gradients_map():
	pass

def produce_heat_map():
	pass


if __name__ == '__main__':
	# load image:
	img = Image.open('statue_of_unity.jpg')
	#turn into gray-scale:


