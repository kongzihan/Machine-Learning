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
        # 将y从类标签转换为每个分类器的二进制值（要么是类i，要么不是类i）
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        # 适用 minimize 函数优化梯度下降
        fmin = minimize(fun=costReg, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x

    return all_theta


# 我们现在准备好最后一步 - 使用训练完毕的分类器预测每个图像的标签。
# 对于这一步，我们将计算每个类的类概率，对于每个训练样本（使用当然的向量化代码），并将输出类标签为具有最高概率的类。
def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    # np.argmax()函数取出h中元素最大值所对应的索引
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax


# ================================== 数据处理 ==================================
# 导入mat数据
dataFile = 'ex3data1.mat'
data = loadmat(dataFile)
# 手写数字，像素数字
# print(data)
# X 为输入数据(5000,400)，y 为输出类别(5000,1)，共 5000 组数据
# print(data['X'].shape)
# print(data['y'].shape)

weightsFile = 'ex3weights.mat'
weightInit = loadmat(weightsFile)
# 初始化的权重参数
# print(weightInit)

# 实现思路
# 首先，我们为theta添加了一个额外的参数（与训练数据一列），以计算截距项（常数项）。
# 其次，我们将y从类标签转换为每个分类器的二进制值（要么是类i，要么不是类i）。
# 最后，我们使用SciPy的较新优化API来最小化每个分类器的代价函数。
# 如果指定的话，API将采用目标函数，初始参数集，优化方法和jacobian（渐变）函数。 然后将优化程序找到的参数分配给参数数组。
# 实现向量化代码的一个更具挑战性的部分是正确地写入所有的矩阵，保证维度正确。

# rows = 5000 ，有 5000 组手写数字数据
rows = data['X'].shape[0]
# print(rows)

# params = 400 ， 每组手写数字有 400 个像素点数据
params = data['X'].shape[1]
# print(params)

# 10个类别，对应10个分类器，所以参数共有 10x401 个
all_theta = np.zeros((10, params + 1))
# 为theta添加了一个额外的参数（与训练数据一列），以计算截距项（常数项）
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
theta = np.zeros(params + 1)

# 将y从类标签转换为每个分类器的二进制值（要么是类i，要么不是类i）
# y_0 表示 若类标签为 0，则其值为 1
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))

all_theta = one_vs_all(data['X'], data['y'], 10, 1)
y_pred = predict_all(data['X'], all_theta)

# 计算准确率
# zip () 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print('accuracy = {0}%'.format(accuracy * 100))
