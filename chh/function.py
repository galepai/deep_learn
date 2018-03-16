

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
                img = cv2.resize(img, (width, height))

            images.append(img)
    images = np.array(images)
    print('read %d images.' % images.shape[0])
    return images

# method
# images = read_image('images')
# for i in range(images.shape[0]):
#     cv2.imshow('image', images[i])
#     cv2.waitKey()


# 待检测的图片路径
def detect_face(cv_image):

    # 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade.load('e:\github\deeplearn\haarcascade_frontalface_default.xml')

    # 灰度度处理时间加快
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # 探测图片中的人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    # scaleFactor=None, minNeighbors=None, flags=None, minSize=None, maxSize=None
    print('find %d faces!' % len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(cv_image, (x, y), (x+w, y+w), (0, 255, 0), 2)
        face = cv_image[y:y+w, x:x+w]
        face = cv2.resize(face, (64, 64))
    # cv2.imshow("Find Faces!", face)
    # cv2.waitKey(0)
    return face, x, y, w, h


