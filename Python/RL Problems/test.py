import random
def breed(p1, p2):
    ret = {}
    l = [p1[i] if random.randrange(0,2) == 0 else p2[i] for i in p1.keys()]
    for m,k in enumerate(p1.keys()):
        ret[k] = l[m]
    return ret

print(breed({'a':1,'b':2,'c':3},{'a':0,'b':0,'c':0}))