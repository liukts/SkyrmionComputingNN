from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)

LW = 2.0
MS = 6
FS = 15

dir = './context/'
trials = ['220314_tskone_context1_lr0.4','220314_tskone_context2_lr0.4','220314_tskone_context3_lr0.4','220314_tskone_context4_lr0.4']

# test loop
figtest,axtest = plt.subplots(1,1,figsize=(6,5))
for fname in trials:
    data = np.load(dir + fname + '/accuracies.npy')
    epochs = np.linspace(1,np.shape(data)[1],np.shape(data)[1])
    axtest.plot(np.mean(data,axis=0),linewidth=LW)
for fname in trials:
    data = np.load(dir + fname + '/accuracies.npy')
    epochs = np.linspace(1,np.shape(data)[1],np.shape(data)[1])
    axtest.fill_between(epochs,np.mean(data,axis=0)+np.std(data,axis=0),np.mean(data,axis=0)-np.std(data,axis=0),alpha=0.3)
axtest.set_ylim([41,105])
axtest.set_xlabel('Epochs',fontsize=FS)
axtest.set_ylabel('Test accuracy (%)',fontsize=FS)
axtest.legend(['Arrangement 1','Arrangement 2','Arrangement 3','Arrangement 4'],fontsize=FS)
axtest.tick_params(labelsize=FS)
plt.show()