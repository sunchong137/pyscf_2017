#!/usr/local/bin/python

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/Users/sunchong/ChanGroup/test_pyscf/ft/data/E-T_2.dat")
T = data[:, 0]
E = data[:, 1]
plt.plot(T, E, linewidth = 1.5)
plt.scatter(T, E, s=10)
plt.xlabel(r"$k_BT$")
plt.ylabel(r"$E$")
plt.xlim(0, 6)
plt.savefig("../E-T-2.png", dpi = 200)
plt.show()
