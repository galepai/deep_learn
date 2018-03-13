from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time
import sys
import numpy as np

restore_train = True
epoch = 1
batch_size = 100
requie_acc = 0.995
# model_dir = './cnn_sum_model'   # cross_entropy with reduce_sum
model_dir = './cnn_model'     # cross_entropy with reduce_mean

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([7, 7, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()  # defaults to saving all variables

sess.run(tf.global_variables_initializer())
Num_Trans = mnist.train.images.shape[0]
Num_Validas = mnist.validation.images.shape[0]
Num_Test = mnist.test.images.shape[0]
print('Tran Image Samples: ', Num_Trans)
print('Valida Image Sample: ', Num_Validas)
print('Test Image Samples:', Num_Test)
print('Start Traning ..........', )

total_time = 0.0
valida_acc_list = []
if not restore_train:
    for i in range(epoch):
        num_batch = Num_Trans//batch_size
        num_start = 0
        print("----------------- Epoch %d ---------------------\n" % (i+1))
        start = time.clock()
        for j in range(num_batch):
            batch = mnist.train.next_batch(batch_size)
            num_start += batch_size
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("Epoch %d/%d: %d/%d, training accuracy %g" % (i+1, epoch, num_start, mnist.train.images.shape[0],  train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        valida_acc = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
        print("valida accuracy: ", valida_acc)
        end = time.clock()
        epoch_time = end - start
        total_time += epoch_time
        print("---------this epoch train time: %.2f s-----------\n" % epoch_time)
        # saver.save(sess, './cnn_sum_model/model.ckpt', global_step=i + 1)
        saver.save(sess, model_dir + '/model.ckpt', global_step=i + 1)
        valida_acc_list.append(valida_acc)
        if valida_acc > requie_acc:
            print('%d epoch acc to %f !' % (i+1, requie_acc))
            break
else:
    # ckpt = tf.train.get_checkpoint_state('./cnn_sum_model')
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('restore model sucess!')
    else:
        sys(0)
    print('continue train ............')
    for i in range(epoch):
        num_batch = Num_Trans // batch_size
        num_start = 0
        print("----------------- Epoch %d ---------------------\n" % (i + 1))
        start = time.clock()
        for j in range(num_batch):
            batch = mnist.train.next_batch(batch_size)
            num_start += batch_size
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("Epoch %d/%d: %d/%d, training accuracy %g" %
                  (i + 1, epoch, num_start, mnist.train.images.shape[0], train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        valida_acc = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
        print("valida accuracy: ", valida_acc)
        end = time.clock()
        epoch_time = end - start
        total_time += epoch_time
        print("---------this epoch train time: %.2f s-----------\n" % epoch_time)
        # saver.save(sess, './cnn_sum_model/model.ckpt', global_step=i + 1)
        saver.save(sess, model_dir + '/model.ckpt', global_step=i + 1)
        valida_acc_list.append(valida_acc)
        if valida_acc > requie_acc:
            print('%d epoch acc to %f !' % (i + 1, requie_acc))
            break

print("Training Complete!")
print('Training time : %.2f s' % total_time)

print('max valida_acc:', np.max(valida_acc_list))
print('min valida_acc:', np.min(valida_acc_list))
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# ckpt = tf.train.get_checkpoint_state('./cnn_model')
# if ckpt and ckpt.model_checkpoint_path:
#     saver.restore(sess, ckpt.model_checkpoint_path)
#     print('restore model sucess!')
# else:
#     sys(0)
#     print('decode mnist ............')
#
#
# _cross_entropy = cross_entropy.eval(feed_dict={x: mnist.test.images[0:101].reshape(-1, 784), y_: mnist.test.labels[0:101], keep_prob: 1.0})
# _y_conv = y_conv.eval(feed_dict={x: mnist.test.images[35:37].reshape(-1, 784), y_: mnist.test.labels[35:37], keep_prob: 1.0})
# _cross_entropy2 = cross_entropy2.eval(feed_dict={x: mnist.test.images[0:101].reshape(-1, 784), y_: mnist.test.labels[0:101], keep_prob: 1.0})
# a = mnist.test.labels[35:37]
#
# print(_cross_entropy)
# print(_cross_entropy2)

