from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False

dvc = 50
delta = 0.05

def m(n):
    k = min(dvc,n)
    m = 0
    for i in range(k+1):
        m += comb(n,k)
    return m

#Original VC-bound
def f1(n):
    e = ((8/n)*np.log(4*m(2*n)/delta)) ** 0.5
    return e


#Rademancher Penalty Bound:
def f2(n):
    e = ((2/n) * np.log(2*n*m(n))) ** 0.5 +(((2/n)*np.log(1/delta)))**0.5 + 1/n
    return e

#Parrondo and Van den Broek:
def f3(n):
    e = ((1/n)*np.log((6*m(2*n)/delta))+1/(n**2))**0.5 + 1/n
    return e

#Devroye:
def f4(n):
    e= (np.log(4*m(n**2)/delta)/(2*(n-2))+1/((n-2)**2))**0.5 + 1/(n-2)
    return e


x = np.arange(100,2000)

y1=[f1(i) for i in x]
y2=[f2(i) for i in x]
y3=[f3(i) for i in x]
y4=[f4(i) for i in x]

plt.plot(x,y1, label = 'Original VC-bound')
plt.plot(x,y2, label = 'Rademancher Penalty Bound')
plt.plot(x,y3, label = 'Parrondo and Van den Broek')
plt.plot(x,y4, label = 'Devroye')
plt.legend()
plt.xlabel("N")
plt.ylabel("epsilon")
plt.show()
