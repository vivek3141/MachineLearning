import numpy as np


def sigmoid(x, d=False):
    if (d == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

arr = []
out = []
test = [1,2,3,4,5,6,7,8,9,1,4,7,2,5,8,3,6,9,1,5,9,3,5,7,1,3,2,4,6,5,7,9,8,1,7,4,2,8,5,3,9,6
,1,9,5,3,7,5]
for i in range(2, len(test), 3):
                        arr.append([1,test[i-2], test[i-1]])
                        out.append(test[i])

print(arr, out)
X = np.array(arr)
y = np.array([out]).T
np.random.seed(1)
syn0 = 2 * np.random.random((3, 1)) - 1
print(syn0)
for i in range(10000):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l1_error = y - l1
    l1_delta = l1_error * sigmoid(l1, True)
    syn0 += np.dot(l0.T, l1_delta)
print("Output After Training:")
print(l1)
