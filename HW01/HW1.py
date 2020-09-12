# HomeWork 1
import cv2
import matplotlib.pyplot as plt
import os


# Implement this function using opencv equalizeHist function
def equalize_hist1(image):
    #Write your code here (part a)
    return


# Implement this function without using opencv equalizeHist function
def equalize_hist2(image):
    #Write your code here (part b)
    return


# main
src_path = 'images/'
dst_path1 = 'results1/'
dst_path2 = 'results2/'
names = os.listdir(src_path)

for name in names:
    # load image
    src_name = src_path + name
    image = cv2.imread(src_name, cv2.IMREAD_GRAYSCALE)

    # equalize histogram and write
    image1 = equalize_hist1(image)
    dst_name1 = dst_path1 + name
    cv2.imwrite(dst_name1, image1)

    # equalize histogram and write
    image2 = equalize_hist2(image)
    dst_name2 = dst_path2 + name
    cv2.imwrite(dst_name2, image2)
