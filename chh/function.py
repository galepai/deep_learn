

import numpy as np
import cv2
import os


def read_images(dir_path, extension='.jpg', height=0, width=0):

    images = []

    for filename in os.listdir(dir_path):
        if filename.endswith(extension):
            filename = dir_path + '/' + filename
            img = cv2.imread(filename)
            if height != 0 or width != 0:
                img = cv2.resize(img, (height, width))

            images.append(img)
    images = np.array(images)

    return images

# method
# images = read_image('images')
# for i in range(images.shape[0]):
#     cv2.imshow('image', images[i])
#     cv2.waitKey()


