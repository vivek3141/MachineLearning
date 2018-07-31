import numpy as np
import random


def fitness(modelList, modelScore, parameterList):
    max1 = np.argmax(modelScore, axis=0)
    a = [0 if i == max1 else i for i in modelScore]
    max2 = np.argmax(a, axis=0)
    ind1 = modelScore.index(max1)
    ind2 = modelScore.index(max2)
    ind3 = random.randrange(0, len(modelScore))
    while ind3 != ind1 and ind3 != ind2:
        ind3 = random.randrange(0,len(modelScore))
    return [modelList[ind1], modelList[ind2], modelList[ind3]], [parameterList[ind1], parameterList[ind2], parameterList[ind3]]


print(fitness([9, 9, 9, 9, 9, 9], [0, 1, 2, 3, 4, 5], [9, 9, 9, 9, 9, 9]))
