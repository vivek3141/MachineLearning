import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a - b * np.exp(c * x)


time = np.array([2.888, 4.463, 8.255, 15.503, 17.817, 35.448, 38.72, 47.926])
avg = np.array([79.64, 43.4, 32.33, 17.99, 15.37, 10.1, 9.8, 9.7])
initialGuess = [5, 5,-.01]
guessedFactors = [func(x, *initialGuess) for x in time]
t = curve_fit(func, time, avg,initialGuess)
cont = np.linspace(min(time),max(time),50)
fittedData = [func(x, *t) for x in cont]
fig = plt.figure(1)
a = fig.add_subplot(1,1,1)
a.plot(time,avg,linestyle='',marker='o', color='r',label="data")
a.plot(cont,fittedData,linestyle='-', color='g',label="model")
a.legend(loc=0, title="legend", fontsize=12)
a.set_ylabel("average")
a.set_xlabel("time")
a.grid()
a.set_title("Chang Hong Lik averages")
plt.show()
