import cv2 as cv
import numpy as np
import os
import time


def gaussian(r2, std=1):
    """
    Sample one instance from gaussian distribution regarding
    given squared-distance:r2, standard-deviation:std and general-constant:k

    :param r: squared distance from center of gaussian distribution
    :param std: standard deviation

    :return: A sampled number obtained from gaussian
    """
    return np.exp(-r2/(2.*std**2)) / (2.*np.pi*std**2)

# implement a function that returns a gaussian filter
def make_gaussian(std=1, size=3):
	"""
    Creates a gaussian kernel regarding given size and std.
    Note that to define interval with respect to the size, 
    I used linear space sampling which may has
    lower accuracy from renowned libraries.

    :param std: standard deviation value
    :param size: size of the output kernel
    :return: A gaussian kernel with size of (size*size)
    """

    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    distance = x**2+ y**2
    kernel = gaussian(r2=distance, std=std)
    return kernel
  
# implement a 2D convolution
def convolve2d(image, kernel):
    """
    Applies 2D convolution via provided kernel over given image

    :param image: input image in grayscale mode
    :param kernel: kernel of convolution
    :return: A convolved image with 'same' size using zero-padding 
    """
    # you do not need to modify these, but feel free to implement however you know from scratch
    kernel       = np.flipud(np.fliplr(kernel))  # Flip the kernel, if it's symmetric it does not matter
    k = kernel.shape[0]
    padding      = (k - 1)
    offset       = padding // 2
    output       = np.zeros_like(image)
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + padding, image.shape[1] + padding))   
    image_padded[offset:-offset, offset:-offset] = image

    # implement the convolution inside the inner for loop
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            output[y,x] = np.sum(kernel * image_padded[y:y+k, x:x+k])
    return output


# use opencv Gaussian blur to complete this function
def opencv_gaussian_blur(image, std, size):
    """
    Applies gaussian filter to smooth image
    
    :param image: Open cv or numpy ndarray image
    :param size: The size of gaussian filter
    :param std: if 0, it will be calculated automatically, otherwise for x and y sould be indicated.
    :return: An open cv image
    """
    return cv2.GaussianBlur(src=image, ksize=(size, size), sigmaX=std, sigmaY=std)


# use opnecv median blur to complete this function
def opencv_median_blur(image, size):
    """
    Applies median filter regarding given window `size`
    
    :param image: open cv or numpy ndarray image
    :param size: size of median kernel
    :return: An open cv image
    """
    median_blured = cv2.medianBlur(image, size)
    return median_blured


src_path = 'images_noisy/'
dst_path = 'images_filtered/'
names = os.listdir(src_path)

# Do not modify these for loops
for name in names:
    # load image
    src_name = src_path + name
    image = cv.imread(src_name, cv.IMREAD_GRAYSCALE)

    # gaussian blur using your implementation
    start = time.time()
    kernel1 = make_gaussian(std=1, size=11)
    blur1 = convolve2d(image, kernel1)
    dt1 = time.time() - start
    dst_name1 = dst_path + name + '.bmp'
    cv.imwrite(dst_name1, blur1)
    
    # gaussian blur using opencv implementation
    start = time.time()
    blur2 = opencv_gaussian_blur(image, std=1, size=11)
    dt2 = time.time() - start
    dst_name2 = dst_path + name + '_CV.bmp'
    cv.imwrite(dst_name2, blur2)
    
    # meidan blur 
    start = time.time()
    blur3 = opencv_median_blur(image)
    dt3 = time.time() - start
    dst_name3 = dst_path + name + '_median.bmp'
    cv.imwrite(dst_name3, blur3)

