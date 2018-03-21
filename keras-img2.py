from chh.function import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import load_model
import keras.callbacks as callbacks
from keras.preprocessing.image import ImageDataGenerator

first_train = False
is_continue_train = True
is_test = not is_continue_train

train_dir = 'e:\Python_Project\images\image'

num_epochs = 1
batch_size = 100

data_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.1)

# classes: 可选参数,为子文件夹的列表,如['dogs','cats']默认为None.
# 若未提供,则该类别列表将从directory下的子文件夹名称/结构自动推断。
# 每一个子文件夹都会被认为是一个新的类。(类别的顺序将按照字母表顺序映射到标签值)。
# 通过属性class_indices可获得文件夹名与类的序号的对应字典。
train_generator = data_gen.flow_from_directory(train_dir,
                                               target_size=(64, 64),
                                               batch_size=batch_size,
                                               class_mode='categorical', subset='training')
validation_generator = data_gen.flow_from_directory(train_dir,
                                               target_size=(64, 64),
                                               batch_size=batch_size,
                                               class_mode='categorical', subset='validation')

# print(train_generator.class_indices)
# validation_generator = data_gen.flow_from_directory(validation_dir,
#                                                     target_size=(64, 64),
#                                                     batch_size=batch_size,
#                                                     class_mode='categorical')

if is_continue_train:

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
        tensor_board = callbacks.TensorBoard()
        model.fit_generator(generator=train_generator,
                            epochs=num_epochs,
                            validation_data=validation_generator,
                            callbacks=[tensor_board])
        model.save('my_model_2.h5')
    else:
        print('continue train .............')
        model = load_model('my_model.h5')
        model.summary()
        tensor_board = callbacks.TensorBoard()
        model.fit_generator(generator=train_generator,
                            epochs=num_epochs,
                            validation_data=validation_generator,
                            callbacks=[tensor_board])
        model.save('my_model_2.h5')
else:
    model = load_model('my_model_2.h5')

if is_test:
    # _img = read_images('e:\Python_Project\images\mix_pic')

    img_path = 'e:\Python_Project\images\own pic\8923.jpg'
    # img_path = 'own.jpg'
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
