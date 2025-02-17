# TensorBoard可视化

import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, n_layer, out_size, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='Weights')
            tf.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1)
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights)+biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data)-0.5 + noise

# 定义placeholder，向网络输入数值
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 添加隐藏层
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# 添加输出层
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# 预测值与真实值之间的误差
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                        reduction_indices=[1]),name='loss')
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#   如果要每隔50次输出一次残差，加下面语句即可
#   if i % 50 == 0:
    print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

# 在项目目录下运行命令 tensorboard --logdir='logs/' 打开网址即可查看

