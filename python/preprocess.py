# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt

def process(source, IMAGE_SIZE=227):
    im = cv2.imread(source)
    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    # plt.imshow(im)
    # plt.show()
    im = im[:, :, ::-1]  # RGB TO BGR
    im = im.transpose((2, 0, 1))  # X1*X2*3 TO 3*X1*X2
    return im

if __name__ == "__main__":
    process("/home/bw/loss-function-comparation/CASIA-WebFace/0000045/001.jpg")

