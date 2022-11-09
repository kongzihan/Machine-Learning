"""
多类分类

这部分，你需要实现一个可以识别手写数字的神经网络。
对于此练习，我们将使用逻辑回归来识别手写数字（0到9）。
我们将扩展我们在练习2中写的逻辑回归的实现，并将其应用于一对一的分类。

神经网络可以表示一些非线性复杂的模型。权重已经预先训练好，你的目标是在现有权重基础上，实现前馈神经网络。
若已给定神经网络中的theta矩阵（需要用反向传播算法得出），实现前馈神经网络，理解神经网络的作用。

题目已给出a(1)为第一层输入层数据，有400个神经元代表每个数字的图像（不加偏置值）；
a(2)为隐藏层，有25个神经元（不加偏置值）；
a(3)为输出层‘，又10个神经元，以10个（0/1）值的向量表示；
theta1为第一层到第二层的参数矩阵（25，401）；
theta2为第二层到第三层的参数矩阵（10，26）。

数据集：
ex3data1.mat文件
输入是图片的像素值，20*20像素的图片有400个输入层单元，不包括需要额外添加的加上常数项。
mat数据格式是matlab的数据存储的标准格式

ex3weights.mat文件
材料已经提供了训练好的神经网络参数 theta1， theta2，有25个隐层单元和10个输出单元（10个输出）

你需要实现前馈神经网络预测手写数字的功能，
和之前的一对多分类一样，神经网络的预测会把（h（x））k中值最大的，作为预测输出

"""
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


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


# 逻辑回归模型的添加正则化的梯度下降函数注意，
# 我们实际上没有在这个函数中执行梯度下降，我们仅仅在计算一个梯度步长。
# 在练习中，由于我们使用Python，我们可以用SciPy的“optimize”命名空间用来优化函数来计算成本和梯度参数
# 以下是原始代码是使用for循环的梯度函数：
def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if i == 0:
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])
    return grad


# 向量化的梯度函数
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    # 截距梯度未正则化
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()


# 预测结果
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# 构建分类器
# 对于这个任务，我们有10个可能的类，并且由于逻辑回归只能一次在2个类之间进行分类，我们需要多类分类的策略。
# 在本练习中，我们的任务是实现一对一全分类方法，
# 其中具有k个不同类的标签就有k个分类器，每个分类器在“类别 i”和“不是 i”之间决定。
# 我们将把分类器训练包含在一个函数中，该函数计算10个分类器中的每个分类器的最终权重，并将权重返回为k X（n + 1）数组，其中n是参数数量。
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k X (n + 1) array for the parameters of each of the k classifiers
    # K个分类器，k=num_labels , 每个分类器的所有参数 params + 1
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    # 插入一列作为 bias
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    # 标签是从1开始的，而非从0开始的
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=costReg, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x

    return all_theta
# ================================== 数据处理 ==================================
# 导入mat数据
dataFile = 'ex3data1.mat'
data = loadmat(dataFile)
weightsFile = 'ex3weights.mat'
weightInit = loadmat(weightsFile)


