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

    #定义存储训练轮数的变量.这个变量不需要计算滑动平均值，所以这里制定为不可训练的变量trainable=False
    #在tf训练神经网络时，一般将代表训练轮数的变量指定为不可训练的参数。
    global_step = tf.Variable(0, trainable=False)

    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    #给定训练轮数的变量可以加速训练早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #在所有代表神经网络参数的变量上滑动平均。
    #其他辅助变量就不需要
    #tf.trainable_variables()返回的就是图上的集合
    #GraphKeys.TRAINABLE_VARIABLES中元素是所有没有指定trainable=False的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    #计算使用量滑动平均之后的前向传播结果。
    #滑动平均不会改变变量本身的取值，而是维护一个影子变量来记录其滑动平均值。
    #当需要使用这个滑动平均值，需要明确调用average函数
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数。
    #这里使用sparse_softmax_cross_entropy_with_logits函数计算交叉熵。
    #当分类问题只有一个正确答案，可以使用这个函数来加速交叉熵的计算。
    #mnist问题只包含0-9中一个数字，所以可以使用这个函数计算交叉熵。
    #这个函数第一个参数是神经网络不包括softmax层的前向传播结果。第二个是训练数据的正确答案。
    #因为标准答案是一个长度10的一维数组。而该函数需要提供的是一个正确答案的数字，
    #所以需要使用tf.arg_max函数来得到正确答案对应的类别编号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    #计算当前batch中所有样例的交叉熵平均值。
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算l2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏执项
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,                     #基础的学习率，随着迭代的进行，更新变量时使用的
        global_step,                            #当前迭代轮数
        mnist.train.num_examples / BATCH_SIZE,  #过完所有的训练数据需要迭代次数
        LEARNING_RATE_DECAY                     #学习率摔衰减速度
    )

    #使用tf.train.GradientDescentOptimizer优化损失函数。
    #这里的损失函数包含交叉熵损失和l2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值。
    #为了一次完成多个操作。tf提供来control_dependencies和group
    #train_op = tf.group(train_step, variable_averages_op)和下面等价的
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    #检验使用来滑动平均模型的神经网络前向传播结果是否正确。
    #
    #
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
