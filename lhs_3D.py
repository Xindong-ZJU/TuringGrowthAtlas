import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import *
import numpy as np

npara = 3
nsample = 100
sunny = lhs(npara, samples = nsample, criterion = "center")

c1_sample = np.array(10**(sunny[:,0]*2 - 1))
c2_sample = np.array(10**(sunny[:,1]*2))
c3_sample = np.array(10**(sunny[:,2]*2))
c4_sample = np.full(nsample,1000)

parameter_sample = np.vstack((c1_sample,c2_sample,c3_sample,c4_sample))
parameter_sample = parameter_sample.transpose()

parameternames = ['c1','c2','c3','c4']
df = pd.DataFrame(data=parameter_sample, columns=parameternames)

print(df)

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(122, projection='3d')

ax.scatter(df['c1'],df['c2'], df['c3'])

ax3 = fig.add_subplot(121)

plt.scatter(df['c1'], df['c2'])
plt.show()

# print(parameter_sample)




