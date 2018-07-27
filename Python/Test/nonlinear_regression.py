import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
def func(x, a, b, c):
    return a - b * np.exp(c * x)
time = np.array([2.888, 4.463, 8.255, 15.503, 17.817, 35.448, 38.72, 47.926])
avg = np.array([79.64, 43.4, 32.33, 17.99, 15.37, 10.1, 9.8, 9.7])
initialGuess = [5, 5, -.01]
guessedFactors=[func(x, *initialGuess) for x in time]
popt,pcov = curve_fit(func, time, avg, initialGuess)
print(popt)
cont=np.linspace(min(time),max(time),50)
fittedData=[func(x, *popt) for x in cont]
fig1 = plt.figure(1)
ax=fig1.add_subplot(1,1,1)
ax.plot(time,avg,linestyle='',marker='o', color='r',label="data")
ax.plot(cont,fittedData,linestyle='-', color='g',label="model")
ax.legend(loc=0, title="legend", fontsize=12)
ax.set_ylabel("average")
ax.set_xlabel("time")
ax.grid()
ax.set_title("Chang Hong Lik averages")
plt.show()
