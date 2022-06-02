from helper import graph_population
from os import walk

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import n_nodes


folder = f"dbs/{n_nodes}_nodes_net"
f = []
d = []
for (dirpath, dirnames, filenames) in walk(folder):
    d.extend(dirnames)
    f.append(filenames)

paths = []
i = 0
stats = []
utilizations = []
histograms = []
db_names = []
for ff in f[i]:
    if ff.find("html") == -1:
        db_names.append(ff)
db_names.sort()
for ff in db_names:
    G = graph_population(folder, ff)
    u = []
    for edge in G.edges:
        u.append(G.edges[edge]["utilization"]/G.edges[edge]["capacity"])
    utilizations.append(u)
    histograms.append(np.hstack(u))

n_plots = len(db_names)

x_plots = 3
y_plots = math.ceil(n_plots/x_plots)

fig, axs = plt.subplots(x_plots, y_plots)
weights = np.ones_like(utilizations[1])/len(utilizations[1])
j = -1
for i in range(0, x_plots*y_plots):
    if i < len(utilizations):
        if i % x_plots == 0:
            j = j+1
        weights = np.ones_like(utilizations[i])/len(utilizations[i])
        axs[i % x_plots, j].hist(utilizations[i], weights=weights)
        axs[i % x_plots, j].set_title(db_names[i])

fig.tight_layout()

plt.show()
