"""Split high-fidelity data into training and testing data."""

import os
import numpy as np
import sys

np.random.seed(0)

internal_points = False
tke_filter = True
#train_sizes = [1e4, 1e5]
#train_sizes = [5000]

print("\nScript:", sys.argv[0])
print("\nData path:", sys.argv[1])
print("\nOut ext:", sys.argv[2])
print("\nTrain size:", sys.argv[3])

data_path = sys.argv[1]
out_ext   = sys.argv[2]
train_sizes = [int(sys.argv[3])]

#data_path = "/scratch/punim0394/SQUAREDCYLINDER/WMSC/02_Training/my_training_data_rnd11"
data_ext = "edf"

out_path = "/scratch/punim0394/SQUAREDCYLINDER/WMSC/02_Training"
#out_ext = "rnd11"

data = {}
for f in os.listdir(data_path):
    if f[-len(data_ext):] == data_ext:
        
        print(f"Load {f}.")
        data[f.replace("T", "V")] = np.loadtxt(os.path.join(data_path, f))

# select internal points
if internal_points:
    internal_size = 3275250
    internal_mask = np.arange(internal_size)

    for var in data:
        print(f"Select internal points of {var}.")
        data[var] = data[var][internal_mask]
        print(f"Reduced shape of {var} is {data[var].shape}.")

# apply TKE filter
if tke_filter:
    tke_var = "k.edf"
    tke_limit = 0.001
    tke_mask = data[tke_var] > tke_limit

    for var in data:
        print(f"Apply TKE filter on {var}.")
        data[var] = data[var][tke_mask]
        print(f"Reduced shape of {var} is {data[var].shape}.")

data_size = data[list(data.keys())[0]].shape[0]

samples = np.arange(data_size)

for t in sorted(train_sizes, reverse=True):

    print(f"Store training data set of size {t:.0f}.")

    samples = np.random.choice(samples, size=int(t), replace=False)
    
    mask = np.full(data_size, False)
    mask[samples] = True

    for f in data.keys():

        print(f"Store {f}.")
        
        # save training data
        train_folder = "training_data_{}_{:.0e}".format(out_ext, t).replace('+0', '')
        if train_folder not in os.listdir():
            os.mkdir(train_folder)
        np.savetxt(os.path.join(out_path, train_folder, f), data[f][mask])

        # save testing data
        test_folder = "testing_data_{}_{:.0e}".format(out_ext, t).replace('+0', '')
        if test_folder not in os.listdir():
            os.mkdir(test_folder)
        np.savetxt(os.path.join(out_path, test_folder, f), data[f][~mask])

