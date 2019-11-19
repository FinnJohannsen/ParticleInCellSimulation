# %% def
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from os import walk


def load_files(path):
    dic = {}
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            dic[filename.split('.')[0]] = np.loadtxt(f"{dirpath}/{filename}", skiprows=1, delimiter=',')
    return dic


def combine(measure):
    return np.concatenate([measure[key] for key in measure])


# %%load data
measure = load_files("data")
data = combine(measure)

# %%calc
U = data[:, 0]
I = data[:, 1]
r = data[:, 2]

I2r2 = I ** 2 * (r * 1E-3) ** 2
lin_reg = stats.linregress(U, I2r2)

R = 150E-3
mu0 = 1.256E-6
n = 130
em = 125 / 32 * (R / (mu0 * n)) ** 2 * 1 / lin_reg.slope
print(f"slope: {lin_reg.slope:.4E} A^2m^2/V")
print(f"intercept: {lin_reg.intercept:.4E} A^2m^2")
print(f"stderr: {lin_reg.stderr:.4E} A^2m^2/V")
print(f"e/m = {em:.4E} C/Kg")
# %% plotting
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel("$U$ in $V$", usetex=True)
ax.set_ylabel(" $I^2r^2 $ in $A^2m^2$", usetex=True)
ax.plot(U, I2r2, "+")
ax.plot(U[[0, -1]], lin_reg.slope * U[[0, -1]] + lin_reg.intercept, "--", linewidth=1)

plt.savefig("results/plot.png")
plt.show()
