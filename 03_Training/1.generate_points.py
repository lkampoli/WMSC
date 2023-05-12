import numpy as np
import os
import sys

try:
    file_path = sys.argv[1]
    file_name = sys.argv[2]
except:
    file_path = "./"
    file_name = "new_points.H"

xmin = 1.5
xmax = 8
xppd = 16
xpoints = int((xmax-xmin)*xppd + 1)
print(xpoints)

ymin = -5
ymax = 0
yppd = 20
ypoints = int((ymax-ymin)*yppd + 1)
print(ypoints)

zmin = 0
zmax = 5
zppd = 35
zpoints = int((zmax-zmin)*zppd + 1)
print(zpoints)

with open(os.path.join(file_path, file_name), 'w') as f:
    for x in np.linspace(xmin, xmax, xpoints):
        for y in np.linspace(ymin, ymax, ypoints):
            for z in np.linspace(zmin, zmax, zpoints):
                point = f"({x} {y} {z})\n"
                f.write(point)
