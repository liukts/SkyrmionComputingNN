import numpy as np
import matplotlib.pyplot as plt
import os

rootdir = './outputs/'
# os.chdir(rootdir)
legend = []
for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        print(dir)
        data = np.load(rootdir + dir + '/accuracies.npy')
        plt.plot(data)
        legend.append(dir.split("_")[1] + "-" + dir.split("_")[2])
        if os.path.isfile(rootdir + dir + '/mnist_accuracies.npy'):
            data = np.load(rootdir + dir + '/mnist_accuracies.npy')
            plt.plot(data)
            legend.append(dir.split("_")[1] + "-" + dir.split("_")[2] + "-mnist")
        if os.path.isfile(rootdir + dir + '/kmnist_accuracies.npy'):
            data = np.load(rootdir + dir + '/kmnist_accuracies.npy')
            plt.plot(data)
            legend.append(dir.split("_")[1] + "-" + dir.split("_")[2] + "-kmnist")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(legend)
plt.title("Multi-task learning test accuracy")
plt.show()
