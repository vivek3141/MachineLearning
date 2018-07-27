import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def create_layer(layer_number, input_size, output_size, x, nonlinear=True):
    with tf.variable_scope(f"layer_{layer_number}"):
        W = tf.get_variable("W", [input_size, output_size])
        b = tf.get_variable("b", [1, output_size])
        y = tf.matmul(x, W) + b
        if nonlinear:
            y = tf.sigmoid(y)
        return y


x = tf.placeholder(tf.float32, shape=[None, 784])
yy = tf.placeholder(tf.float32, shape=[None, 10])
y = create_layer(0, 784, 10, x, False)

J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=yy, logits=y))
optim = tf.train.MomentumOptimizer(0.01, 0.001)
min = optim.minimize(J)

s = tf.Session()
s.run(tf.global_variables_initializer())

testy = s.run(J, feed_dict={x: mnist.test.images, yy: mnist.test.labels})
print(testy)

losses = []
for i in range(1000):
    xval, yval = mnist.train.next_batch(100)
    if i % 50:
        losses.append(s.run(J, feed_dict={x: mnist.train.images[502].reshape(1, 784), yy: mnist.train.labels[502].reshape(1, 10)}))
    a = s.run(min, feed_dict={x: xval, yy: yval})

plt.plot(range(len(losses)), losses)
plt.show()

testy = s.run(J, feed_dict={x: mnist.test.images, yy: mnist.test.labels})
print(testy)

a = cv2.imread("test.png")
gray_image = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
gray = (gray_image.reshape(1, 784))/255
im1 = (gray_image.reshape(784,1))/255
print(im1)
two_d = (np.reshape(im1, (28, 28)) * 255).astype(np.uint8)
cv2.imshow("a", two_d)
cv2.waitKey(0)

ypred = s.run(y, feed_dict={x: gray})

print("Predictions:", ypred)