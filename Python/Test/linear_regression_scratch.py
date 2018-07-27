import numpy as np
import matplotlib.pyplot as plt

plt.ylabel("Y Values")
plt.xlabel("X Values")


def linear_regression(X, y, m_current=0, b_current=0, epochs=1000, learning_rate=0.0001):
    N = float(len(y))
    for i in range(epochs):
        y_current = (m_current * X) + b_current
        cost = sum([data ** 2 for data in (y - y_current)]) / N
        m_gradient = -(2 / N) * sum(X * (y - y_current))
        b_gradient = -(2 / N) * sum(y - y_current)
        m_current = m_current - (learning_rate * m_gradient)
        b_current = b_current - (learning_rate * b_gradient)
    return m_current, b_current, cost


def matrixsub(mat1, mat2):
    mat = []
    for i in range(len(mat1)):
        mat.append(mat1[i] - mat2[i])
    return mat


def costfunction(theta, x, y):
    j = 1 / len(x)
    sigma = 0
    for i in range(len(x)):
        sigma = sigma + (y[i] - hypothesis(theta, x[i]))
    j = j * sigma
    return j


def gradient(x, y, theta, n):
    weight = (2 / n) * (-x) * (y - hypothesis(theta, x))
    bias = (2 / n) * (y - hypothesis(theta, x))
    return [weight, bias]


def hypothesis(theta, x):
    return theta[0] * x + theta[1]


theta = [1, 1]
x = np.array([1, 2, 1, 4, 2, 5, 4, 6, 3, 1, 3, 5, 2, 6, 4, 6, 4, 6, 7, 8, 2, 8, 9, 2, 6, 3, 4, 7, 8, 2, 9, 5, 4, 6, 3, 2, 7, 3,
     8, 9])
y = np.array([1, 3, 1, 6, 8, 3, 7, 4, 7, 5, 3, 1, 8, 2, 9, 0, 4, 6, 3, 2, 3, 4, 3, 8, 9, 4, 7, 4, 2, 8, 9, 0, 0, 7, 4, 2, 6, 4,
     7, 2])
n = len(x)
for i in range(len(x)):
    theta = matrixsub(theta, gradient(x[i], y[i], theta, n))
plt.scatter(x, y, c="RED")
m,b,cost = linear_regression(x,y)
theta[0] = m
theta[1] = b
xLine = np.array(range(0, 10))
yLine = theta[0] * xLine + theta[1]
plt.plot(xLine, yLine)
plt.show()
plt.close()
