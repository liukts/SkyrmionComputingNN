from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)

LW = 2.0
MS = 6
FS = 15

dir = './context/'
trials = ['220428_wcancer_ann_direct','0920_tskone_context_lr0.5_dummy4']
colors = ['tab:purple','mediumaquamarine']

# test loop
figtest,axtest = plt.subplots(1,1,figsize=(6,4))
for idx,fname in enumerate(trials):
    data = np.load(dir + fname + '/accuracies.npy')
    epochs = np.linspace(1,np.shape(data)[1],np.shape(data)[1])
    axtest.plot(np.mean(data,axis=0),linewidth=LW,color=colors[idx])
for idx,fname in enumerate(trials):
    data = np.load(dir + fname + '/accuracies.npy')
    epochs = np.linspace(1,np.shape(data)[1],np.shape(data)[1])
    axtest.fill_between(epochs,np.mean(data,axis=0)+np.std(data,axis=0),np.mean(data,axis=0)-np.std(data,axis=0),alpha=0.3,color=colors[idx])
axtest.set_ylim([41,105])
axtest.set_xlabel('Epochs',fontsize=FS)
axtest.set_ylabel('Test accuracy (%)',fontsize=FS)
axtest.set_ylim([50,105])
axtest.legend(['Software (direct)','Context (34 neurons)'],fontsize=FS)
axtest.tick_params(labelsize=FS)
plt.show()