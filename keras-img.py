from chh.function import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

own_images = read_images('images')
other_images = read_images('pics')
images = np.append(own_images, other_images)
_images = images.reshape((-1, 64, 64, 3))


own_labels = np_utils.to_categorical(np.random.randint(1, 2, 100), 2)
other_labels = np_utils.to_categorical(np.random.randint(0, 1, 100), 2)
labels = np.append(own_labels, other_labels)
_labels = labels.reshape(-1, 2)
train_x, test_x, train_y, test_y = train_test_split(_images, _labels, test_size=0.1, random_state=np.random.randint(0, 100))
train_x = train_x / 255.0
test_x = test_x / 255.0

# 建模
model = Sequential()

# 卷积层，对二维输入进行滑动窗卷积
# 当使用该层为第一层时，应提供input_shape参数，在tf模式中，通道维位于第三个位置
# border_mode：边界模式，为"valid","same"或"full"，即图像外的边缘点是补0
# 还是补成相同像素，或者是补1, 本例中input_shape为(28, 28, 1),1表示图像为1通道
model.add(Convolution2D(32, 3, 3,
                        border_mode='same',
                        input_shape=(64, 64, 3)))
model.add(Activation('relu'))


# 卷积层，激活函数是ReLu
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

# 池化层，选用Maxpooling，给定pool_size，dropout比例为0.25
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten层，把多维输入进行一维化，常用在卷积层到全连接层的过渡
model.add(Flatten())

# 包含128个神经元的全连接层，激活函数为ReLu，dropout比例为0.5
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 包含10个神经元的输出层，激活函数为Softmax
model.add(Dense(2))
model.add(Activation('softmax'))

# plot_model(model, to_file='model.png')
# 输出模型的参数信息
model.summary()
# 配置模型的学习过程
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(x=train_x, y=train_y, batch_size=32, epochs=50, verbose=1, validation_data=(test_x, test_y))

preds = model.evaluate(x=test_x, y=test_y)
# score = model.evaluate(test, Y_test, verbose=0)
print(preds)

score = model.evaluate(x=own_images[14:20], y=own_labels[14:20])
print(score)

