

import numpy as np
import cv2
import os


# 读取路径下的所有文件
def read_images(dir_path, extension='.jpg', height=0, width=0):
    """"
         Examples
        --------
            images = read_images('images')
            for i in range(images.shape[0]):
                cv2.imshow('image', images[i])
                cv2.waitKey()

    """
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


# 检测的人脸
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
        # cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = cv_image[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
    # cv2.imshow("Find Faces!", face)
    # cv2.waitKey(0)
    return face, x, y, w, h


# 找两张图像的特征点，生成齐次矩阵，im_aligned对齐im_reference
def align_images(im_aligned, im_reference, max_features=500, good_match_percent=0.15):
    """
        Examples
        --------
          # Read reference image
          refFilename = "im_reference.jpg"
          print("Reading reference image : ", refFilename)
          imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

          # Read image to be aligned
          imFilename = "im_aligned.jpg"
          print("Reading image to align : ", imFilename);
          im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

          print("Aligning images ...")
          # Registered image will be resotred in imReg.
          # The estimated homography will be stored in h.
          im_align, imMatches, h = alignImages(im, imReference)
          cv2.imwrite('match images', imMatches)

          # Write aligned image to disk.
          outFilename = "aligned.jpg"
          print("Saving aligned image : ", outFilename);
          cv2.imwrite(outFilename, imReg)

          # Print estimated homography
          print("Estimated homography : \n",  h)

        """

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im_aligned, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im_reference, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im_aligned, keypoints1, im_reference, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im_reference.shape
    im_align = cv2.warpPerspective(im_aligned, h, (width, height))

    return im_align, imMatches, h


