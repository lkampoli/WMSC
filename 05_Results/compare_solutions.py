import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

# Reference data solutions 
##########################

# Fabian's solutions
####################
path = '01_Data/postProcessing/sets/20059.95/'

# Velocity profiles 
X3D_U       = pd.read_csv(path+'X3D_U.csv')
X3D_k_Rterm = pd.read_csv(path+'X3D_k_Rterm.csv')
X3D_aij     = pd.read_csv(path+'X3D_aij.csv')

print(X3D_U)
print(X3D_k_Rterm)

plt.figure(figsize=(30,15), frameon=False)
plt.scatter(X3D_U['U_0']+0, X3D_U['z'], label='x/d=3')
#
plt.xlabel('u_i/u_f [-]')
plt.ylabel('z/d [-]')
#
plt.grid()
plt.legend()
plt.savefig('velocity.png')
#plt.show()
plt.close()


plt.figure(figsize=(30,15), frameon=False)
plt.scatter(X3D_k_Rterm['k']+0, X3D_k_Rterm['k'], label='x/d=3')
#
plt.xlabel('k/u_f^2 [-]')
plt.ylabel('z/d [-]')
#
plt.grid()
plt.legend()
plt.savefig('TKE.png')
#plt.show()
plt.close()
