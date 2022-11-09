"""
你将建立一个逻辑回归模型来预测一个学生是否被大学录取。
假设你是一所大学系的管理员，你想根据两次考试的成绩来决定每个申请人的录取机会。
您有以前申请者的历史数据，可以用作逻辑回归的训练集。
对于每个培训示例，您都有申请人在两次考试中的分数和录取决定。
你的任务是建立一个分类模型，根据这两次考试的分数来估计申请人的录取概率。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


# ================================函数定义部分====================================
# 逻辑函数Sigmoid，在公式中g(z)表示,g(z)=g(theta.T* X)=h(x)
def sigmoid(z):
    return 1/(1+np.exp(-z))


# 代价函数
def computeCost(theta, X, y):
    # print(X.shape)
    # print(y.shape)
    # print(theta.shape)
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1-y), np.log(1-sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


# 梯度下降(未使用这个函数)
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(theta, X, y)
        # print(theta.shape)
    return theta, cost


# 计算一个梯度的步长
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad


# 预测录取结果
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# ================================== 数据处理 ==================================
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['grade-1', 'grade-2', 'y/n'])
# 打印前10行数据
# print(data.head(10))

# 打印数据信息
# print(data.describe())

# 绘制散点图
# data.plot(kind='scatter', x='grade-1', y='y/n', figsize=(12, 8))
# plt.show()
positive = data[data['y/n'].isin([1])]
negative = data[data['y/n'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['grade-1'], positive['grade-2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['grade-1'], negative['grade-2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
# 根据可视化的结果，可以看到有线性决策边界

# 在训练集中添加一列
data.insert(0, 'ones', 1)

# 设置x和y的值
cols = data.shape[1]  # shape[0]输出行数，shape[1]输出列数
X = data.iloc[:, 0:cols - 1]  # X是所有行，去掉最后一列
y = data.iloc[:, cols - 1:cols]  # y是最后一列

X = np.matrix(X.values)
y = np.matrix(y.values)
# theta 是一个(1,3)矩阵
theta = np.matrix(np.array([0, 0, 0]))

# 计算代价函数 (theta初始值为0).

loss1 = computeCost(theta, X, y)
# print("loss1 is %.*f" % (3, loss1))

# alpha = 0.01
# iters = 1000
# g, cost = gradientDescent(X, y, theta, alpha, iters)
# loss2 = computeCost(X, y, g)
# print("loss2 is %.*f" % (3, loss2))

# 现在可以用SciPy's truncated newton（TNC）实现寻找最优参数。
result = opt.fmin_tnc(func=computeCost, x0=theta, fprime=gradient, args=(X, y))
# print(result)
computeCost(result[0], X, y)
# print(computeCost(result[0], X, y))

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
# map将correct中的数据映射为int型
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
