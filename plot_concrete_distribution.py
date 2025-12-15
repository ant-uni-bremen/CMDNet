#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:35:17 2019

@author: beck
"""

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tplt



# Concrete distribution plot and its relation to MAP
from scipy.stats import norm, laplace
alpha = np.array([0.5, 0.5])
tau = 0.1
n_res = 200
x_lim = [-0.1, 1.1]

x0 = np.linspace(x_lim[0], x_lim[1], n_res)[np.newaxis, :]
n = 2
x = np.concatenate((x0, 1 - x0), axis=0) #np.repeat(x[np.newaxis, :], n, axis = 0)
# Faktor 0.5 wegen Transformation der Zufallsvariablen
p_x = 0.5 * np.math.factorial(n - 1) * tau ** (n - 1) * np.prod(alpha[:, np.newaxis] * x ** (- tau -1), axis=0) / np.sum(alpha[:, np.newaxis] * x ** (- tau), axis=0) ** n
x2 = np.linspace(2 * x_lim[0] - 1, 2 * x_lim[1] - 1, n_res)
N_x = norm.pdf(2 * (x2 - 0.4), 0, 1)
#L_x = laplace.pdf(x2, 0, 1)
nlnp = -np.log(p_x) - np.log(N_x)
plt.figure(1)
plt.plot(x2, -np.log(p_x), 'r--', label='-ln p(x)')
plt.plot(x2, -np.log(N_x), 'k--', label='-ln p(y|x)')
plt.plot(x2, nlnp, 'b-', label='-ln p(x,y)')
#plt.plot(x2, -np.log(L_x / np.max(N_x) * np.nanmax(p_x)), 'r-', label='Laplace')
plt.grid()
plt.xlabel("x")
plt.ylabel("-ln p(x,y)")
plt.legend()
#plt.show()
tplt.save("plots/concrete_distribution.tikz")


# Concrete distribution plot
tau = 2
alpha_bin = 0.5
p_x = 0.5 * np.math.factorial(n - 1) * tau ** (n - 1) * np.prod(alpha[:, np.newaxis] * x ** (- tau - 1), axis=0) / np.sum(alpha[:, np.newaxis] * x ** (- tau), axis=0) ** n 
p_xbin = 0.5 * tau * alpha_bin * x0 ** (- tau - 1) * (1 - alpha_bin) * (1 - x0) ** (- tau - 1) / (alpha_bin * x0 ** (- tau) + (1 - alpha_bin) * (1 - x0) ** (- tau)) ** 2
plt.figure(2)
plt.plot(x2, p_x, 'r-', label='p(x) concrete relax')
plt.plot(np.array([-1, 1]), alpha, 'bo', label='p(x) true')
plt.plot(x2, p_xbin[0,:], 'k-', label='p(x) binary concrete relax')
plt.grid()
plt.xlabel("x")
plt.ylabel("p(x)")
#plt.ylim(0, 2)
plt.legend()
#plt.show()
tplt.save("plots/concrete_distribution_vs_bernoulli.tikz")

