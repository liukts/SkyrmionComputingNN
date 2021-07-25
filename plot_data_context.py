import numpy as np
import matplotlib.pyplot as plt
import os

rootdir = './outputs/context/'
# os.chdir(rootdir)
legend = []
for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        print(dir)
        data = np.load(rootdir + dir + '/accuracies.npy')
        plt.plot(data,'.-')
        legend.append(dir.split("_")[2])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(legend)
plt.title("T-SKONE training")
plt.show()