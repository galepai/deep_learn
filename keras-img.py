from chh.function import *

images = read_images('images', height=360, width=480)
print(images.shape)
cv2.imshow('image', images[0])
cv2.waitKey()
