# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 20:45:24 2019

@author: saimunikoti
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cp

#%% checking relationship between parameter and optimal value

# Generate data.
m = 20
n = 15

A = np.random.randn(m, n)
b = np.random.randn(m)
result = np.zeros((10,2))

# Define and solve the CVXPY problem.
x = cp.Variable(n)
cost = cp.sum_squares(A*x - b)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()


# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)

#%% exponential problem
np.random.seed(1)
a = 2
x = np.linspace(-5,5,100)
y = np.exp(1*x)
ypow = np.power(x-a,2)
plt.plot(x,ypow, color="cornflowerblue")

#%%
def solveproblem(a):
    
    x1 = cp.Variable(1)
    x2 = cp.Variable(1)
    
    # Form objective.
    obj = cp.Minimize((4*(x1 -x2- a))**2)
    constraints = [x1 <= 4]
              
    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    
    print("optimal var", x1.value)
    x1 = x1.value
    x2 = x2.value
    return x1[0] , x2[0]  

xopt=[]  

for count in range(10):
    xopt.append(solveproblem(count))
    
    
#%% 
    
t=np.zeros((1000,3))

x1 = np.linspace(-8,4,1000)
x2 = np.linspace(-8,4,1000)

for count,x1ind in enumerate(x1):
    for x2ind in x2:
         t[count,0]=1*(x1ind**2) + 1*(x2ind) - (x1ind*x2ind)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1,x2,t[:,0],'lightseagreen')       


