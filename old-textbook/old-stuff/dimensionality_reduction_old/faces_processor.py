import os
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
instances = 500
faces = people.data[:instances, :]
print(faces.shape)

##root = os.getcwd()
##os.chdir(root + '/faces')
##files = os.listdir()
##x = []
##
##
##for filename in files:
##    if not(filename == ".DS_Store"):
##        x.append(scipy.ndimage.imread(filename))
##
faces -= faces.mean()
faces /= faces.var()**0.5
np.save('dim_red_faces.npy', faces)
