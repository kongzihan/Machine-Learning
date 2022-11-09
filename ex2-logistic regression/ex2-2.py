"""
正则化逻辑回归在练习的这一部分，
您将实施正则化逻辑回归来预测制造工厂的微芯片是否通过质量保证。
在质量保证过程中，每个微芯片都要经过各种测试，以确保其功能正常。
假设你是工厂的产品经理，你有两个不同测试的一些微芯片的测试结果。
从这两个测试中，你可以决定微芯片应该被接受还是被拒绝。
为了帮助您做出决定，您有一个过去微芯片的测试结果数据集，从中可以构建逻辑回归模型
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model  # 调用sklearn的线性回归包


# ==================================== 函数定义部分 =========================================
# 逻辑函数Sigmoid，在公式中g(z)表示,g(z)=g(theta.T* X)=h(x)
def sigmoid(z):
    return 1/(1+np.exp(-z))


# 逻辑回归模型的添加正则化的代价函数
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X*theta.T)))
    reg = (learningRate / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2)))
    return np.sum(first - second) / (len(X)) + reg


# 逻辑回归模型的添加正则化的梯度下降函数
def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])

    return grad


# 预测结果
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# ================================== 数据处理 ==================================
path = 'ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test-1', 'Test-2', 'y/n'])
# 打印前10行数据
# print(data.head(10))

# 打印数据信息
# print(data.describe())

# 绘制散点图
positive = data2[data2['y/n'].isin([1])]
negative = data2[data2['y/n'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test-1'], positive['Test-2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test-1'], negative['Test-2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
# 根据可视化的结果，可以看到没有线性决策边界

# 创建一组多项式特征
degree = 5
x1 = data2['Test-1']
x2 = data2['Test-2']
# 插入bias列
data2.insert(3, 'Ones', 1)
# print(data2.head())
# 添加多项式列
for i in range(1, degree):
    for j in range(0, i):
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
# 删掉 test-1 和 test-2 两列
data2.drop('Test-1', axis=1, inplace=True)
data2.drop('Test-2', axis=1, inplace=True)
# print(data2.head())

# 设置 x 和 y 的值
cols = data2.shape[1]
X2 = data2.iloc[:, 1:cols]
y2 = data2.iloc[:, 0:1]

# convert to numpy arrays and initalize the parameter array theta
# 转化为 np.array 并初始化参数
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(11)
# 设置 λ
learningRate = 1
# 计算参数为0时的cost
costReg(theta2, X2, y2, learningRate)


# 计算一次梯度下降的步长
gradientReg(theta2, X2, y2, learningRate)
# 计算优化后的结果
result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
theta_min = np.matrix(result2[0])
# 计算预测结果
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) / len(correct))
print('accuracy = {0}%'.format(accuracy))


# 调包的方法
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X2, y2.ravel())
print(model.score(X2, y2))
