
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_ratio_list = []
train_epoch_list = []

def createData(dataNum, w, b, sigma):
    train_x = np.arange(dataNum)
    train_y = w * train_x + b + np.random.randn() * sigma
    # train_y = w * train_x + b
    print(train_x)
    print(train_y)
    return train_x, train_y


def linerRegression(train_x, train_y, epoch=50000, rate=0.00000005):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    n = train_x.shape[0]
    x = tf.placeholder("float")
    y = tf.placeholder("float")
    w = tf.Variable(tf.random_normal([1]))  # 生成随机权重
    b = tf.Variable(tf.random_normal([1]))

    pred = tf.add(tf.multiply(x, w), b)
    loss = tf.reduce_sum(tf.pow(pred - y, 2))
    # loss = tf.reduce_sum(tf.abs(pred - y))
    # cross_entropy = -tf.reduce_sum(y * tf.log(pred + 1e-10))
    # loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)
    # optimizer = tf.train.AdamOptimizer().minimize(loss)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    print('w  start is ', sess.run(w))
    print('b start is ', sess.run(b))
    saver = tf.train.Saver()
    tf.summary.FileWriter('logfile', sess.graph)
    for index in range(epoch):
        # for tx,ty in zip(train_x,train_y):
        # sess.run(optimizer,{x:tx,y:ty})
        _, _loss, _w, _b = sess.run([optimizer, loss, w, b], {x: train_x, y: train_y})
        if index % 5000 == 0:
            print('epoch ', index)
            # print('pred is ', sess.run(pred, {x: train_x}))
            print('loss is ', _loss)
            train_ratio_list.append(_loss)
            train_epoch_list.append(index)
            print('w is ', _w)
            print('b is ', _b)
            print('------------------')
            saver.save(sess, './model/model.ckpt', global_step=epoch + 1)
    print('loss is ', sess.run(loss, {x: train_x, y: train_y}))
    w = sess.run(w)
    b = sess.run(b)
    return w, b


def predictionTest(test_x, test_y, w, b):
    W = tf.placeholder(tf.float32)
    B = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    n = test_x.shape[0]
    pred = tf.add(tf.multiply(X, W), B)
    loss = tf.reduce_mean(tf.pow(pred - Y, 2))
    sess = tf.Session()
    loss = sess.run(loss, {X: test_x, Y: test_y, W: w, B: b})
    return loss


if __name__ == "__main__":
    train_x, train_y = createData(50, 2.0, 10.0, 1.0)
    test_x, test_y = createData(20, 2.0, 10.0, 1.0)
    w, b = linerRegression(train_x, train_y)
    print('weights', w)
    print('bias', b)
    # loss = predictionTest(test_x, test_y, w, b)
    # print(loss)


    # plt.figure()
    plt.subplot(221)
    plt.scatter(train_x, train_y)
    plt.plot([train_x[0], train_x[train_x.size-1]], [w*train_x[0]+b, w*train_x[train_x.size-1]+b], 'r')
    plt.subplot(222)
    plt.scatter(test_x, test_y)
    plt.plot([test_x[0], test_x[test_x.size - 1]], [w * test_x[0] + b, w * test_x[test_x.size - 1] + b], 'r')
    plt.subplot(223)
    plt.plot(train_epoch_list[5:], train_ratio_list[5:], 'g')
    plt.show()
