import numpy as np
import matplotlib.pyplot as plt

#input data
a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y_data=[0.0026, 0.0123, -0.0052, -0.0026, -0.0017, 0, 0.0035, 0.0052, 0.0009, -0.0035, 0, 0.0052]
b_data = np.matrix([0.0026, 0.0123, -0.0052, -0.0026, -0.0017, 0, 0.0035, 0.0052, 0.0009, -0.0035, 0, 0.0052])

#polynomial regression where n=3
b_data=b_data.T
x_data=[]
for i in a_data:
    x_data.append([1,i,i**2,i**3,i**4,i**5])
x_data=np.matrix(x_data)
# print(x_data)
def reg(b_data,x_data):
    theta=np.dot(np.dot(np.linalg.inv(np.dot(x_data.T,x_data)),x_data.T),b_data)
    return theta

def cal_CPI(x_data,theta):
    cpi=np.array(np.dot(x_data,theta))
    # print(cpi)
    CPI=[]
    for i in cpi:
        CPI.append(i[0])
    return CPI


#LOOCV
def LOOCV(a_data,b_data,x_data,y_data):
    k=0
    J=0
    while k<=5:
        xnew=np.delete(x_data,k,0)
        bnew=np.delete(b_data,k,0)
        # print(bnew)
        theta=reg(bnew,xnew)
        # print(theta)
        CPI=cal_CPI(x_data,theta)
        # print(CPI)
        # print(np.power(np.dot(x_data[k],theta),2))
        # print(b_data[k])
        J += 1/12*(J_func(theta,x_data[k],b_data[k]))
        # print(J)
        plot_curves(a_data,y_data,CPI,k)
        k += 1
    return J




def plot_curves(a_data,y_data,CPI,k):
    plt.scatter(a_data, y_data, s=50)
    plt.plot(a_data,CPI)
    plt.scatter(a_data[k],y_data[k],color='r')
    plt.xlabel("Times")
    plt.ylabel("CPI")
    plt.show()

def J_func(theta, x_data, b_data):
    J = 0
    J = np.power(np.dot(x_data,theta)-b_data,2)
    return J



def predict(t,b_data,x_data):
    x=np.matrix([1,t,t**2,t**3,t**4,t**5])
    theta=reg(b_data,x_data)
    CPI=cal_CPI(x,theta)
    return CPI

J=LOOCV(a_data,b_data,x_data,y_data)
theta=reg(b_data,x_data)
CPI_Jan=predict(13,b_data,x_data)
CPI_Feb=predict(14,b_data,x_data)

# print(theta)
# print(CPI_Jan)
# print(CPI_Feb)
# print(J)

