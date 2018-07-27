import matplotlib as plt
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
img = (mnist.train.images[0]+1) * 255
#im1 = mnist.train.images[1]
a = cv2.imread("test.png")
gray_image = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
im1 = 1-(gray_image.reshape(784, 1))/255
print(im1)
two_d = (np.reshape(im1, (28, 28)) * 255).astype(np.uint8)
cv2.imshow("a", two_d)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(mnist.train.labels[1])
for i in range(0,28):
    for k in im1[28*i:28*i+28]:
        temp = 0
        if k != 0:
            temp = 1
        print(temp, end="")
    print("")
