import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
# pandas.csv() 函数将逗号分离的值 （csv） 文件读入数据框架
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 打印数据前几行,默认输出前5行
# print(data.head(10))
# describe会返回一系列参数，count，mean，std，min，25%，50%，75%，max
# print(data.describe())

# scatter是散点图
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()


# 计算代价函数，其实就是误差的平方和/2m,m为个数
def computeCost(X, y, theta):
    # power（）函数用于计算几次方
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# 在训练集中添加一列
# 添加ones列是为了使矩阵乘法的结果为1*b+xi*w
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]  # shape[0]输出行数，shape[1]输出列数
# iloc[0:2,:]是取前两行
# iloc[:,0:2]是取前两列
X = data.iloc[:, 0:cols - 1]  # X是所有行，去掉最后一列
y = data.iloc[:, cols - 1:cols]  # y是最后一列

X = np.matrix(X.values)
y = np.matrix(y.values)
# theta 是一个(1,2)矩阵
theta = np.matrix(np.array([0, 0]))

# 计算代价函数 (theta初始值为0).
loss1 = computeCost(X, y, theta)
print("loss1 is %.*f" % (3, loss1))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    # ravel()方法将数组维度拉成一维数组,shape[0]输出行数，shape[1]输出列数
    # parameters为参数个数
    parameters = int(theta.ravel().shape[1])
    # cost[]记录每次迭代的代价cost
    cost = np.zeros(iters)

    # 训练1000次
    for i in range(iters):
        error = (X * theta.T) - y

        # 需要注意的是，w和b两个参数需要同时更新，所以使用temp临时变量进行整体更新
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


alpha = 0.01
iters = 1000
g, cost = gradientDescent(X, y, theta, alpha, iters)
loss2 = computeCost(X, y, g)
print("loss2 is %.*f" % (3, loss2))

# numpy.linspace()函数用于在线性空间中以均匀步长生成数字序列。
x = np.linspace(data.Population.min(), data.Population.max(), 100)
# y=b+wx
f = g[0, 0] + (g[0, 1] * x)

# 线性拟合
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)  # 设置图例的位置
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

# 代价-迭代次数图
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(np.arange(iters), cost, 'r')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Cost')
ax1.set_title('Error vs. Training Epoch')
plt.show()
