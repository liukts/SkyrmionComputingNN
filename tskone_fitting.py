import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_func(x, a, b, c):
    # Curve fitting function
    return a * x**3 + b * x**2 + c*x # d=0 is implied

input = np.array([3.00,3.25,3.50,3.75,4.00,4.25,4.50])
input_ext = np.array([-4.5,-4.25,-4.0,-3.75,-3.5,-3.25,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,-0.1,0,0.1,0.5,1.0,1.5,2.0,2.5,3.0,3.25,3.50,3.75,4.0,4.25,4.5])
config1_ext = np.array([-1.202,-1.191,-1.180,-1.123,-1.053,-1.024,-0.996,-0.798,-0.628,-0.499,-0.283,-0.142,-0.029,0,0.029,0.142,0.283,0.499,0.628,0.798,0.996,1.024,1.053,1.123,1.180,1.191,1.202])
config1 = np.array([0.996,1.024,1.053,1.123,1.180,1.191,1.202])
config2 = np.array([0.740,0.799,0.834,0.895,0.941,0.967,0.990])

degree = 3
c1poly = np.polyfit(input,config1,degree)
params = curve_fit(fit_func,input_ext,config1_ext)
c1poly_ext = params[0]
c2poly = np.polyfit(input,config2,degree)

x = np.linspace(3,4.5,151)
x_ext = np.linspace(-4.5,4.5,901)
c1x = 0
c1x_ext = 0
c2x = 0
for i in range(0,degree):
    c1x += c1poly[i]*x**(degree-i)
    c1x_ext += c1poly_ext[i]*x_ext**(degree-i)
    c2x += c2poly[i]*x**(degree-i)

#plt.plot(input,config1)
plt.plot(input,config2,'.-')
plt.plot(input_ext,config1_ext,'.-')
#plt.plot(x,c1x)
plt.plot(x_ext,abs(c1x_ext))
plt.plot(x,c2x)
plt.show()

np.save('./poly/c1poly.npy',c1poly_ext)
np.save('./poly/c2poly.npy',c2poly)