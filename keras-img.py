from chh.function import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import keras.callbacks as callbacks
from sklearn.model_selection import train_test_split

first_train = False
is_continue_train = False
is_test = True

num_epochs = 1
batch_size = 100

if is_continue_train:
    own_images = read_images('e:\Python_Project\images\my_faces')
    other_images = read_images('e:\Python_Project\images\other_faces')
    images = np.append(own_images, other_images)
    _images = images.reshape((-1, 64, 64, 3))

    own_labels = np_utils.to_categorical(np.random.randint(1, 2, own_images.shape[0]), 2)
    other_labels = np_utils.to_categorical(np.random.randint(0, 1, other_images.shape[0]), 2)
    labels = np.append(own_labels, other_labels)
    _labels = labels.reshape(-1, 2)
    train_x, test_x, train_y, test_y = train_test_split(_images, _labels, test_size=0.1, random_state=np.random.randint(0, 100))
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    if first_train:
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
        tensorboard = callbacks.TensorBoard()
        model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=(test_x, test_y), callbacks=[tensorboard])
        model.save('my_model.h5')
    else:
        print('continue train .............')
        model = load_model('my_model.h5')
        tensorboard = callbacks.TensorBoard()
        model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=(test_x, test_y), callbacks=[tensorboard])
        model.save('my_model.h5')
else:
    model = load_model('my_model.h5')

if is_test:
    # _img = read_images('e:\Python_Project\images\mix_pic')

    # img_path = 'e:\Python_Project\images\own pic\892.jpg'
    img_path = 'own.jpg'
    image = cv2.imread(img_path)
    face, x, y, w, h = detect_face(image)

    _img = np.expand_dims(face, axis=0)
    result = model.predict(_img)
    # score = model.evaluate(test, Y_test, verbose=0)
    _result = np.argmax(result, axis=1)
    for i in range(_img.shape[0]):
        if _result[i]:
            print('own pic')
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, 'chenhui', (x, y), font, 1, (0, 255, 0), 1)
            cv2.rectangle(image, (x, y), (x + w, y + w), (0, 255, 0), 2)
            cv2.imshow('image', image)
            cv2.waitKey()
        else:
            print('other pic')
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, 'other', (0, 30), font, 1, (0, 255, 0), 1)
            cv2.imshow('image', image)
            cv2.waitKey()

