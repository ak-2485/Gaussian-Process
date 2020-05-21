import numpy as np
from gp import GaussianProcessRegressor
import matplotlib.pyplot as plt

"""
Verify the fit of a GP. 
Plot the Likelihood function.
"""
import sys
sys.path.insert(1, '../')

dim =1
N = 100
a = -1
b = 1
X = np.random.uniform(a,b,(N,dim))
f = lambda x: np.exp(-(x-0.5)**2)*np.sin(30*x)
#f = lambda x: -np.exp(-100*(x-0.8)**2)+np.exp(-2*(x-1)**2)+np.exp(-2*(x+1.5)**2)
fX = f(X).flatten();

# fit the GP
GP = GaussianProcessRegressor()
GP.fit(X,fX)



M  = 400
xx = np.linspace(a,b,M)
m,v = GP.predict(xx.reshape((M,dim)), True)
m  = m.flatten()
v  = v.flatten()

# Plot a GP
plt.plot(xx,m,label='GP')
plt.fill_between(xx,m,m+v,color='b',alpha=0.5)
plt.fill_between(xx,m,m-v,color='b',alpha=0.5)
plt.scatter(X.flatten(),fX,color='k',label='data')
plt.legend()
plt.show()


