import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#mnist数据集相关的常数。
#输入层节点数。对于mnist数据集，这个值为图片的像数
INPUT_NODE = 784
#输出层节点数。这个等于类别的数目。区分的为0-9数字，所以为10
OUTPUT_NODE = 10

#配置神经网络的参数
#隐藏层节点数。这里为只有一个隐藏层的网络结构。这个隐藏层值、有500个节点
LAYER1_NODE = 500

#一个训练batch中的训练数据个数。数字越小时。训练过程越接近。
#随机梯度下降；数字越大，训练越接近梯度下降
BATCH_SIZE = 100

#基础的学习率
LEARNING_RATE_BASE = 0.8
#学习率的衰减率
LEARNING_RATE_DECAY = 0.99

#描述模型复杂度的正则化项在损失韩式中的系数
REGULARIZATION_RATE = 0.0001
#训练轮回数
TRAINING_STEPS = 30000
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

#一个辅助函数。给定神经网络的输入和所有参数，计算神经网络的前向传播结果。
#定义一个使用relu激活函数的三层全连接神经网络。通过加入隐藏层实现多层网络结构
#通过relu激活函数实现去线性化。在这个函数中也支持出入用于计算参数平均值的类
#方便测试时候s
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    #没有提供滑动平均类时，直接使用参数当前取值
    if avg_class == None:
        #计算隐藏层的前向传播结果，这里使用relu激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        #计算输出层的前向传播结果。在计算损失函数时会一并计算softmax。
        #所以这里不需要加入激活函数，而且不加入softmax不影响预测结果。
        #因为预测时使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后的计算没有影响。
        #于是在计算整个神经网络的前向传播时可以不加入最后的softmax层
        return tf.matmul(layer1, weights2) + biases2
    else:
        #首先使用avg_class.average函数计算出变量的滑动平均值
        #然后计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2))+avg_class.average(biases2)

#训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')

    #生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    #生成输出层的参数
    weights2 =tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    #计算在当前参数下的神经网络前向传播结果。这里是用于计算滑动平均的类为None，所以参数不使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    #定义存储训练轮数的变量
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    #
    #一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        test_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s) , validation accuracy using average model is %g" % (i,validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s),test accuracy using average modle is %g" % (TRAINING_STEPS, test_acc)  )

#主程序入口
def main(argv=None):
    #声明出来mnist数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

#tf提供的一个主程序入口。tf.app.run()会调用上面的main函数
if __name__ == '__main__':
    tf.app.run()
