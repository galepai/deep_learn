#coding:utf-8
'''
线性层的softmax回归模型识别手写字
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#mnist数据输入
mnist = input_data.read_data_sets("/home/fly/TensorFlow/mnist", one_hot=True)

x = tf.placeholder("float", [None, 784]) #placeholder是一个占位符，None表示此张量的第一个维度可以是任何长度的

#
w = tf.Variable(tf.zeros([784,10])) #定义w维度是:[784,10],初始值是0
b = tf.Variable(tf.zeros([10])) # 定义b维度是:[10],初始值是0

#
y = tf.nn.softmax(tf.matmul(x, w) + b)

# loss
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) #用 tf.log 计算 y 的每个元素的对数。接下来，我们把 y_ 的每一个元素和 tf.log(y) 的对应元素相乘。最后，用 tf.reduce_sum 计算张量的所有元素的总和。

# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化
init = tf.initialize_all_variables()

# Session
sess = tf.Session()
sess.run(init)

# 迭代
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 50 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        _correct_prediction, _accuracy = sess.run([correct_prediction,accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("Setp: ", i, "Correct_Prediction: ",_correct_prediction, "    Accuracy: ", _accuracy)
        # print("Setp: ", i, "Correct_Prediction: ", sess.run([correct_prediction,accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        # print("Setp: ", i, "Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
