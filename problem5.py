import numpy as np
import matplotlib.pyplot as plt

a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
b_data =np.array([0.26, 1.23, -0.52, -0.26, -0.17, 0, 0.35, 0.52, 0.09, -0.35, 0, 0.52])
# b_data =np.array([26, 123, -52, -26, -17, 0, 35, 52, 9, -35, 0, 52])
x2=[]
for i in a_data:
    x2.append(i**2)
x3=[]
for i in a_data:
    x3.append(i**3)

#initialization
lr=0.0001 # learning rate
theta0 = 0
theta1 = 0
theta2 = 0
theta3 = 0




#least squares formulation
def f_func(theta0, theta1, theta2, theta3, a_data, b_data):
    f = 0
    for i in range(0, len(a_data)):
        f += np.longdouble(((theta0 + theta1 * a_data[i]+ theta2 * x2[i] + theta3 * x3[i])-b_data[i]) ** 2)
    error = np.longdouble(f / float(len(a_data)))
    return error



#gradient descent method
def GD(theta0, theta1, theta2, theta3, a_data, b_data, lr):
    # itera = 0
    # temp=[]
    f= f_func(theta0, theta1, theta2, theta3, a_data, b_data)
    # temp.append([itera, f])
    #begin iteration
    Dx0 = 1
    Dx1 = 1
    Dx2 = 1
    Dx3 = 1
    m=float(len(a_data))
    while np.abs(max(Dx1, Dx2, Dx3)) > 0.0001:#Termination condition
        #compute sum of gradients for each x
        for j in range(0,len(a_data)):
            Dx0 += np.longdouble((2/m) * (theta1 * a_data[j]+ theta2 * x2[j]+ theta3 *x3[j] + theta0) - b_data[j])
            Dx1 += np.longdouble(a_data[j] * (2/m)*((theta1 * a_data[j]+ theta2 *x2[j] + theta3 * x3[j] + theta0) - b_data[j]))
            Dx2 += np.longdouble((x2[j]) * (2/m)*((theta1 * a_data[j]+ theta2 * x2[j]+ theta3 * x3[j] + theta0) - b_data[j]))
            Dx3 += np.longdouble((x3[j]) * (2/m)*((theta1 * a_data[j]+ theta2 * x2[j]+ theta3 * x3[j]+ theta0) - b_data[j]))
        #construct the update rule
        theta0 = np.longdouble(theta0 - lr * Dx0)
        theta1 = np.longdouble(theta1 - lr * Dx1)
        theta2 = np.longdouble(theta2 - lr * Dx2)
        theta3 = np.longdouble(theta3 - lr * Dx3)
        # itera += 1
        f =np.longdouble( f_func(theta0, theta1, theta2, theta3, a_data, b_data))
        # temp.append([itera, f])
    return theta0, theta1, theta2, theta3
    # np.array(temp)



# #plot iteration process
# def plot_f(temp):
#     fig = plt.figure()
#     x=temp[:,0]
#     y=temp[:,1]
#     plt.plot(x,y)
#     plt.title("5-a:iteration process")
#     plt.xlabel("iteration time")
#     plt.ylabel("f")
#     plt.show()

# problem 5(a)
theta0,theta1, theta2, theta3= GD(theta0,theta1, theta2, theta3, a_data, b_data, lr)
# plot_f(temp)
print(theta0,theta1, theta2, theta3)
