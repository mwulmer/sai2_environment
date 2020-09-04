import numpy as np

R = np.zeros((3,3))
t = np.array([[1,1,1]])
X = np.concatenate((R,t.T),axis=1)
X = np.random.normal(loc=0.0, scale=5, size=(3,1))
print(X)