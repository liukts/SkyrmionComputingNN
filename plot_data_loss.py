import numpy as np
import matplotlib.pyplot as plt

dir = './context/'
base = np.load(dir + '0725_tskone_baseline/mean_losses.npy')
direct = np.load(dir + '0725_tskone_direct/mean_losses.npy')
cont = np.load(dir + '0725_tskone_context/mean_losses.npy')

base_avg = np.mean(base,axis=0)
base_err = np.std(base,axis=0)
direct_avg = np.mean(direct,axis=0)
direct_err = np.std(direct,axis=0)
cont_avg = np.mean(cont,axis=0)
cont_err = np.std(cont,axis=0)

epochs = np.linspace(1,len(base_avg),len(base_avg))
plt.plot(base_avg,color='tab:blue')
plt.fill_between(epochs,base_avg-base_err,base_avg+base_err,color='tab:blue',alpha=0.3)
plt.plot(direct_avg,color='tab:orange')
plt.fill_between(epochs,direct_avg-direct_err,direct_avg+direct_err,color='tab:orange',alpha=0.3)
plt.plot(cont_avg,color='tab:green')
plt.fill_between(epochs,cont_avg-cont_err,cont_avg+cont_err,color='tab:green',alpha=0.3)
plt.legend(['Baseline','Direct','Contextual'])
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.grid()
plt.show()