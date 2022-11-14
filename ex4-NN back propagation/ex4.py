"""
题目：
使用神经网络算法识别数据集中的手写数字
数据集包含有：数字集与初始 theta 值

步骤：
构建神经网络模型——初始化向量——向前传播算法——计算代价函数——反向传播，
计算偏导数项——（梯度检验）——高级优化算法下降梯度得到预测值theta——对比预测数据得出准确率。

使用反向传播的前馈神经网络
通过反向传播算法实现神经网络cost函数和梯度计算的非正则化和正则化版本
以及随机权重初始化和使用网络进行预测的方法
"""

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize


# ==================================== 函数定义部分 =========================================
# 逻辑函数Sigmoid，在公式中g(z)表示,g(z)=g(theta.T* X)=h(x)
def sigmoid(z):
    return 1/(1+np.exp(-z))


# 前向传播函数
#  (400 + 1) -> (25 + 1) -> (10)
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    # X.shape = (5000,400) ,插入一列用于和 bias 作乘积
    # a1.shape = (5000,401)
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    # theta1.shape = (25,401)
    # z2.shape = (5000,25)
    z2 = a1 * theta1.T
    # sigmoid z2 之后，再插入一列
    # a2.shape = (5000,26)
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    # theta2.shape = (10,26)
    # z3.shape = (5000,10)
    z3 = a2 * theta2.T
    # sigmoid z3
    h = sigmoid(z3)
    # 输出 h.shape=(5000,10),10个单位对应我们的一个one-hot编码类标签。
    return a1, z2, a2, z3, h


# 正则化的 cost 函数
# def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
#     m = X.shape[0]
#     X = np.matrix(X)
#     y = np.matrix(y)
#
#     # reshape the parameter array into parameter matrices for each layer
#     theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
#     theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
#
#     # run the feed-forward pass
#     a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
#
#     # compute the cost
#     J = 0
#     for i in range(m):
#         first_term = np.multiply(-y[i, :], np.log(h[i, :]))
#         second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
#         J += np.sum(first_term - second_term)
#
#     J = J / m
#
#     # add the cost regularization term
#     J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
#
#     return J


# 接下来是反向传播算法。
# 反向传播参数更新计算将减少训练数据上的网络误差。
# 我们需要的第一件事是计算我们之前创建的Sigmoid函数的梯度的函数。
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# 现在我们准备好实施反向传播来计算梯度。
# 由于反向传播所需的计算是代价函数中所需的计算过程，我们实际上将扩展代价函数以执行反向传播并返回代价和梯度。
# 反向传播计算的最难的部分（除了理解为什么我们正在做所有这些计算）是获得正确矩阵维度。
# 顺便说一下，你容易混淆了A * B与np.multiply（A，B）使用。 基本上前者是矩阵乘法，后者是元素乘法（除非A或B是标量值，在这种情况下没关系）。
def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    # m为输入x的个数
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    # 将参数数组重塑为每个层的参数矩阵
    # (25,401)
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    # (10,26)
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    # 运行前馈神经网络
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    # 初始化
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    # 计算cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    # 添加cost正则化
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # perform backpropagation
    # 反向传播，m = 5000
    # 赋值单个训练集的输入值、激活量、输出值、误差值
    for t in range(m):
        # 第 t 个输入层
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)

        # 第 t 个隐藏层
        a2t = a2[t, :]  # (1, 26)

        # 第 t 个输出层
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        # 计算每一层的误差值
        # 得到输出层的误差
        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # 反向传播算法z也需要加偏置值
        # 得到隐藏层的误差
        d2t = np.multiply((d3t*theta2), sigmoid_gradient(z2t))  # (1, 26)

        # 得到误差矩阵
        delta1 = delta1 + (d2t[:, 1:]).T * a1t  # 去除第一行 （25，401）
        delta2 = delta2 + d3t.T * a2t  # d3t是结果集的误差值，不需要去除第一行，delta2计算为（10，26）

    # 对误差矩阵求均值
    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    # 添加梯度正则化项，并去掉 bias
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    # 将梯度矩阵分解为单个数组
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    print(grad.shape)

    return J, grad

# ================================== 数据处理 ==================================
# 导入mat数据

# 加载手写数字的数据
data = loadmat('ex4data1.mat')
# print(data)
X = data['X']
y = data['y']
# (5000,400) (5000,1)
# print(X.shape, y.shape)

# 对y标签进行 one-hot 编码
# one-hot 编码将类标签n（k类）转换为长度为k的向量，其中索引n为“hot”（1），而其余为0。
# 即one-hot的shape为 y的元素数 x 类别数 = 5000 x 10
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
# print(y_onehot)
# (5000,10)
# print(y_onehot.shape)

# 加载权重数据
# weight = loadmat('ex4weights.mat')
# print(weight)
# theta1 = weight['Theta1']
# theta2 = weight['Theta2']
# (25,401) (10,26)
# print(theta1.shape, theta2.shape)

# 我们要为此练习构建的神经网络具有与我们的实例数据
# （400 + 偏置单元）大小匹配的输入层，25个单位的隐藏层（带有偏置单元的26个），
# 以及一个输出层， 10个单位对应我们的一个one-hot编码类标签。

# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# 随机初始化完整网络参数大小的参数数组
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
# print(params.shape)

# minimize the objective function
# 利用优化函数，得到权重
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
# print(h[0])
# #取出 h 每一行中元素最大值所对应的索引，+1
y_pred = np.array(np.argmax(h, axis=1) + 1)
# print(y_pred)
# 若预测值和真实值相同，则预测正确
# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))
