import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tushare
import os
import datetime

# hidden layer units
rnn_unit = 10
lstm_layers = 2
input_size = 6
output_size = 1
lr = 0.0006
ticker = input('请输入A股股票代码')


def stock_price_intraday(folder):
    intraday = tushare.get_hist_data(ticker, start='2005-1-1', end='2019-1-1')
    temp_file = folder + '/' + ticker + '.csv'
    if os.path.exists(file):
        history = pd.read_csv(file, index_col=0)
        intraday.append(history)
    intraday.sort_index(inplace=True)
    intraday.index.name = 'timestamp'
    intraday.to_csv(temp_file)
    print('intraday for [', ticker, '] got.')


tickers_raw_data = tushare.get_stock_basics()
tickers = tickers_raw_data.index.tolist()
dateToday = datetime.datetime.today().strftime('%Y%m%d')
file = 'C:/Users/cacho/Desktop/python_work/M1569/data/TickerList_' + dateToday + '.csv'
tickers_raw_data.to_csv(file)

stock_price_intraday(folder='C:/Users/cacho/Desktop/python_work/M1569/data/')

f = open('C:/Users/cacho/Desktop/python_work/M1569/data/' + ticker + '.csv')  # 重新写入数据位置和名称等
df = pd.read_csv(f)  # 读入股票数据
data = df.iloc[:, 1:8].values  # 取第1-7列


def get_train_data(batch_size=60, time_step=5, train_begin=0, train_end=600):  # 函数传值
    batch_index = []
    data_train = data[train_begin:train_end]  # 训练数据开始至结束
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 定义标准化语句
    train_x, train_y = [], []  # 训练集
    length = len(normalized_train_data) - time_step
    for i in range(length):  # 以下即是获取训练集并进行标准化，并返回该函数的返回值
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :6]  # 即为前7列为输入维度数据
        y = normalized_train_data[i:i + time_step, 6, np.newaxis]  # 最后一列标签为Y，可以说出是要预测的，并与之比较，反向求参
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


def get_test_data(time_step=5, test_begin=300):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step-1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :6]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 6]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(size) * time_step:, :6]).tolist())
    test_y.extend((normalized_test_data[(size) * time_step:, 6]).tolist())
    return mean, std, test_x, test_y


weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


def lstm_cell():
    # basicLstm单元
    basic_lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basic_lstm, output_keep_prob=keep_prob)
    return basic_lstm


def lstm(x):
    batch_size = tf.shape(x)[0]
    time_step = tf.shape(x)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(x, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for i in range(lstm_layers)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


def train_lstm(batch_size=60, time_step=5, train_begin=0, train_end=600):
    x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred,_ = lstm(x)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练10000次
        for i in range(10000):
            for step in range(len(batch_index) - 1):
                _,loss_ = sess.run([train_op, loss], feed_dict={x: train_x[batch_index[step]:batch_index[step + 1]],
                                                                y: train_y[batch_index[step]:batch_index[step + 1]],
                                                                keep_prob: 0.5})
            print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ", saver.save(sess, 'C:/Users/cacho/Desktop/python_work/M1569/model_save/model.ckpt'))
        print("The train has finished")


# train_lstm()


def prediction(time_step=5):
    x = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred,_ = lstm(x)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_save')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={x: [test_x[step]], keep_prob: 1})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[6] + mean[6]
        test_predict = np.array(test_predict) * std[6] + mean[6]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
        print("The accuracy of this predict:", acc)
        # 以折线图表示结果

        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()


prediction()
