#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./wcancer_data.csv")
x = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)  # prune unused data
diag = {"M": 1, "B": 0}
y = data["diagnosis"].replace(diag)
#%%
# sex: female
cancer_alpha = 211
cancer_nalpha = 1
ncancer_alpha = 211
ncancer_nalpha = 146
sex = []
for i in range(569):
    if y[i]:
        if cancer_alpha:
            sex.append("F")
            cancer_alpha -= 1
        elif cancer_nalpha:
            sex.append("M")
            cancer_nalpha -= 1
    else:
        if ncancer_alpha:
            sex.append("F")
            ncancer_alpha -= 1
        elif ncancer_nalpha:
            sex.append("M")
            ncancer_nalpha -= 1
x1 = x.assign(sex = sex)
#%%
# age >= 50
cancer_alpha = 160
cancer_nalpha = 52
ncancer_alpha = 13
ncancer_nalpha = 344
age_50 = []
for i in range(569):
    if y[i]:
        if cancer_alpha:
            age_50.append("T")
            cancer_alpha -= 1
        elif cancer_nalpha:
            age_50.append("F")
            cancer_nalpha -= 1
    else:
        if ncancer_alpha:
            age_50.append("T")
            ncancer_alpha -= 1
        elif ncancer_nalpha:
            age_50.append("F")
            ncancer_nalpha -= 1
x2 = x1.assign(age_50 = age_50)

#%%
# BMI > 30
cancer_alpha = 147
cancer_nalpha = 65
ncancer_alpha = 98
ncancer_nalpha = 259
bmi_30 = []
for i in range(569):
    if y[i]:
        if cancer_alpha:
            bmi_30.append("T")
            cancer_alpha -= 1
        elif cancer_nalpha:
            bmi_30.append("F")
            cancer_nalpha -= 1
    else:
        if ncancer_alpha:
            bmi_30.append("T")
            ncancer_alpha -= 1
        elif ncancer_nalpha:
            bmi_30.append("F")
            ncancer_nalpha -= 1
x3 = x2.assign(bmi_30 = bmi_30)

#%%
# alcohol intake high
cancer_alpha = 181
cancer_nalpha = 31
ncancer_alpha = 272
ncancer_nalpha = 85
alcohol_high = []
for i in range(569):
    if y[i]:
        if cancer_alpha:
            alcohol_high.append("T")
            cancer_alpha -= 1
        elif cancer_nalpha:
            alcohol_high.append("F")
            cancer_nalpha -= 1
    else:
        if ncancer_alpha:
            alcohol_high.append("T")
            ncancer_alpha -= 1
        elif ncancer_nalpha:
            alcohol_high.append("F")
            ncancer_nalpha -= 1
x4 = x3.assign(alcohol_high = alcohol_high)
# %%
x5 = x4.assign(diagnosis = y)
# %%
x5.to_csv("wcancer_data_context.csv", index = False)