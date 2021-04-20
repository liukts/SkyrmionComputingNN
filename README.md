# SkyrmionComputingNN

Repository for applying T-SKONE neuron on an SNN task. 

The following requirements are necessary to run the programs in the repo:
 - Norse: https://github.com/electronicvisions/norse
 - Breast cancer image dataset: https://www.kaggle.com/paultimothymooney/breast-histopathology-images
      - Create a directory named "cancer" and extract images into it
      - Delete the last directory (the one that is named oddly)
      - Run cancer_dataloader.py to split data into training, testing datasets
