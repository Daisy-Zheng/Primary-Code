# import numpy as np
# import matplotlib.pyplot as plt
# x_data=[]
# a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# for i in a_data:
#     x_data.append([1,i])
# x_data=np.matrix(x_data)
# # print(x_data)
# theta=np.matrix([1,1])
# # b_data =np.array([0.26, 1.23, -0.52, -0.26, -0.17, 0, 0.35, 0.52, 0.09, -0.35, 0, 0.52])
# b_data =np.matrix([0.0026, 0.0123, -0.0052, -0.0026, -0.0017, 0, 0.0035, 0.0052, 0.0009, -0.0035, 0, 0.0052])
# b_data=b_data.T
#
# #initialization
# lr=0.0001 # learning rate
#
#
# #least squares formulation
# def J_func(theta, x_data, b_data):
#     J = 0
#     J = 1/2 * (x_data*theta.T-b_data)*((x_data*theta.T-b_data).T)
#     return J
#
# def GD(theta, x_data, b_data, lr):
#     # J= J_func(theta, x_data, b_data)
#     DJ=np.matrix([1,1])
#     while np.linalg.norm(DJ) > 0.00001:#Termination condition
#         #compute sum of gradients for each x
#         DJ=(x_data*theta.T-b_data).T*x_data
#         #construct the update rule
#         theta = theta - lr * DJ
#         # J =J_func(theta, x_data, b_data)
#         # temp.append([itera, f])
#     return theta
#
# theta= GD(theta, x_data, b_data, lr)
# print(theta)

#
# import numpy as np
# import matplotlib.pyplot as plt
# x_data=[]
# a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# i=0
# for i in range(0,12):
#     x_data.append(1)
#     i+=1
# x_data=(np.matrix(x_data)).T
# # print(x_data)
# theta=1
# # b_data =np.array([0.26, 1.23, -0.52, -0.26, -0.17, 0, 0.35, 0.52, 0.09, -0.35, 0, 0.52])
# b_data =np.matrix([0.0026, 0.0123, -0.0052, -0.0026, -0.0017, 0, 0.0035, 0.0052, 0.0009, -0.0035, 0, 0.0052])
# b_data=b_data.T
#
# #initialization
# lr=0.0001 # learning rate
#
#
# #least squares formulation
# def J_func(theta, x_data, b_data):
#     J = 0
#     J = 1/2 * (x_data*theta-b_data)*((x_data*theta-b_data).T)
#     return J
#
# def GD(theta, x_data, b_data, lr):
#     # J= J_func(theta, x_data, b_data)
#     DJ=1
#     while DJ > 0.00001:#Termination condition
#         #compute sum of gradients for each x
#         DJ=(x_data*theta-b_data).T*x_data
#         #construct the update rule
#         theta = theta - lr * DJ
#         # J =J_func(theta, x_data, b_data)
#         # temp.append([itera, f])
#     return theta
#
# theta= GD(theta, x_data, b_data, lr)
# print(theta)

#
# import numpy as np
# import matplotlib.pyplot as plt
# x_data=[]
# a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# for i in a_data:
#     x_data.append([1,i,i**2])
# x_data=np.matrix(x_data)
# # print(x_data)
# theta=np.matrix([0,0,0])
# # b_data =np.array([0.26, 1.23, -0.52, -0.26, -0.17, 0, 0.35, 0.52, 0.09, -0.35, 0, 0.52])
# b_data =np.matrix([0.0026, 0.0123, -0.0052, -0.0026, -0.0017, 0, 0.0035, 0.0052, 0.0009, -0.0035, 0, 0.0052])
# b_data=b_data.T
#
#
# #initialization
# lr=0.00001 # learning rate


#least squares formulation
# def J_func(theta, x_data, b_data):
#     J = 0
#     J = 1/2 * (x_data*theta.T-b_data)*((x_data*theta.T-b_data).T)
#     return J
#
# def GD(theta, x_data, b_data, lr):
#     # J= J_func(theta, x_data, b_data)
#     DJ=(x_data*theta.T-b_data).T*x_data
#     while np.linalg.norm(DJ) > 0.00001:#Termination condition
#         #compute sum of gradients for each x
#         DJ=(x_data*theta.T-b_data).T*x_data
#         #construct the update rule
#         theta = theta - lr * DJ
#         # J =J_func(theta, x_data, b_data)
#         # temp.append([itera, f])
#     return theta
#
# theta= GD(theta, x_data, b_data, lr)
# print(theta)

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
x_data=[]
a_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for i in a_data:
    x_data.append([1,i,i**2,i**3])

x_data=np.matrix(x_data)
# print(x_data)
b_data = np.matrix([0.0026, 0.0123, -0.0052, -0.0026, -0.0017, 0, 0.0035, 0.0052, 0.0009, -0.0035, 0, 0.0052])
b_data=b_data.T
    # [0.0026, 0.0123, -0.0052, -0.0026, -0.0017, 0, 0.0035, 0.0052, 0.0009, -0.0035, 0, 0.0052]
# b_data=np.array(b_data).reshape(len(b_data),1)
 # np.matrix([0.0026, 0.0123, -0.0052, -0.0026, -0.0017, 0, 0.0035, 0.0052, 0.0009, -0.0035, 0, 0.0052])




# def augmentMatrix(A, b):
#     if(len(A) != len(b)):
#         raise 'The number of rows is different'
#     result = []
#     for i in range(len(A)):
#         row = []
#         for j in range(len(A[i])):
#             row.append(A[i][j])
#         for j in range(len(b[i])):
#             row.append(b[i][j])
#         result.append(row)
#     return result
# B=augmentMatrix(x_data,b_data)

# print(np.linalg.matrix_rank(B))
B=(np.linalg.inv(x_data.T*x_data)*x_data.T*b_data)
print(B)
# x = solve(x_data, b_data)
# print(x_data)
# theta=np.matrix([0,0,0,0])
# # b_data =np.array([0.26, 1.23, -0.52, -0.26, -0.17, 0, 0.35, 0.52, 0.09, -0.35, 0, 0.52])
# b_data =np.matrix([0.0026, 0.0123, -0.0052, -0.0026, -0.0017, 0, 0.0035, 0.0052, 0.0009, -0.0035, 0, 0.0052])
# b_data=b_data.T
#
#
# #initialization
# lr=0.0000001 # learning rate
#
# def J_func(theta, x_data, b_data):
#     J = 0
#     J = 1/2 * (x_data*theta.T-b_data)*((x_data*theta.T-b_data).T)
#     return J
#
# def GD(theta, x_data, b_data, lr):
#     # J= J_func(theta, x_data, b_data)
#     DJ=(x_data*theta.T-b_data).T*x_data
#     while np.linalg.norm(DJ) > 0.00001:#Termination condition
#         #compute sum of gradients for each x
#         DJ=(x_data*theta.T-b_data).T*x_data
#         #construct the update rule
#         theta = theta - lr * DJ
#         # J =J_func(theta, x_data, b_data)
#         # temp.append([itera, f])
#     return theta
#
# theta= GD(theta, x_data, b_data, lr)
# print(theta)
