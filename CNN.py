from __future__ import print_function  
import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data  
import numpy as np  
import matplotlib.pyplot as plt
  
  
#选去1到10的数字  
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  
  
def add_layer(inputs, in_size, out_size, activation_function=None):  
    W = tf.Variable(tf.random_normal([in_size, out_size]))  
    b = tf.Variable(tf.zeros([1,out_size])+0.1)  
    Wb = tf.matmul(inputs, W)+b  
    if activation_function is None:  
        outputs = Wb  
    else:  
        outputs = activation_function(Wb)  
    return outputs  
  
def compute_accuracy_CNN(v_xs, v_ys):  
    global  prediction_CNN  
    y_pre = sess.run(prediction_CNN, feed_dict = {xs:v_xs, keep_prob:1})  
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
    result = sess.run(accuracy, feed_dict = {xs: v_xs, ys:v_ys, keep_prob:1})  
    return result  
  
def kernel_variable(shape):  
    initial = tf.truncated_normal(shape=shape, stddev = 0.1)  
    return tf.Variable(initial)  
  
def bias_variable(shape):  
    initial = tf.constant(0.1, shape=shape)  
    return tf.Variable(initial)  
  
def conv2d(x,W):  
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')  #x为输入，W为卷积权重，strides为滤波器移动范围，取步长为1
  
def max_pool_2x2(x):  
    return tf.nn.max_pool(x, ksize= [1,2,2,1], strides=[1,2,2,1], padding='SAME')  
  
#占位操作
xs = tf.placeholder(tf.float32, [None, 784]) #输入信息，输入向量为1*784的矩阵
ys = tf.placeholder(tf.float32, [None, 10])  #类别标签共有10个类别
keep_prob = tf.placeholder(tf.float32)  
x_image = tf.reshape(xs, [-1,28,28,1])  #将xs转化为28*28的形式
  
# 卷积层
w_conv1 = kernel_variable([7,7,1,32])   #核 5*5, 入层大小 1, 出层大小 32  
b_conv1 = bias_variable([32])  
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)  #输出大小 28*28*32  
h_pool1 = max_pool_2x2(h_conv1)         #输出大小 14*14*32  
  
# 卷积层2
w_conv2 = kernel_variable([7,7,32,64])  #核 5*5, 入层大小 32, 出层大小 64  
b_conv2 = bias_variable([64])  
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+ b_conv2) #输出大小 14*14*64  
h_pool2 = max_pool_2x2(h_conv2)         #输出大小 7*7*64  
  
# 全连接层1
w_fc1 = kernel_variable([7*7*64, 1024])  
b_fc1 = bias_variable([1024])  
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  
  
# 全连接层2
w_fc2 = kernel_variable([1024,10])  
b_fc2 = bias_variable([10])  
prediction_CNN = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)  
  
  
#对数据进行预测，784个节点10个分类
prediction = add_layer(xs, 784, 10, activation_function= tf.nn.softmax)  
  
#计算预测误差
cross_entropy_CNN = tf.reduce_mean(-tf.reduce_sum(ys* tf.log(prediction_CNN), reduction_indices=[1]))
  
train_step_CNN = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_CNN)  
saver = tf.train.Saver()
 
#结果输出 ，循环1500次
Total_test_loss_CNN = np.zeros((int(1501/100)+1), float)  
Total_test_acc_CNN = np.zeros((int(1501/100)+1), float)  
count =0  
with tf.Session() as sess:  
    if int((tf.__version__).split('.')[1]) <12 and int((tf.__version__).split('.')[0])<1:  
        init = tf.initialize_all_veriables()  
    else:  
        init = tf.global_variables_initializer()  
    print(tf.__version__)  
    sess. run(init)  
    print('类型   ','      迭代次数   ','      百分比   ')
        
    for i in range(1501):  
        batch_xs, batch_ys = mnist.train.next_batch(100)  
        sess.run(train_step_CNN, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})  
  
        if i%100 ==0:  
            Total_test_acc_CNN[count] = compute_accuracy_CNN(mnist.test.images, mnist.test.labels)  
            print('正确率:        ', i,'        ', Total_test_acc_CNN[count])  
            loss_CNN = sess.run(cross_entropy_CNN,  
                                feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0})  
            print('损失:          ', i,'        ', loss_CNN)  
            Total_test_loss_CNN[count] = loss_CNN  
            count += 1  
 
#数据存储与绘图
    plt.figure(1, figsize=(12, 4))  
    plt.subplot(121)  
    plt.ylabel('Losses')  
    plt.plot(Total_test_loss_CNN, 'b*-', lw=2)  
  
    plt.subplot(122)  
    plt.ylabel('Accuracy:')  
    plt.plot(Total_test_acc_CNN, 'b*-', lw=2)  
    plt.show() 