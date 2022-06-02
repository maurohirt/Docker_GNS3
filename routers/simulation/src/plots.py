#!/usr/bin/env python3
#
# Plot of average loss rate for different algorithms
# ToDo: plot the loss rate in function of the number of flows
#

import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = 'output/20_nodes_net_SW/'

algs = ['OSPF', 'Weighted', 'Threshold']

alg_data = []

for alg in algs:
    res = f'{INPUT_DIR}{alg.lower()}.json'

    f = open(res)
    data = json.loads(f.read())

    for i in range(len(data['flows'])):
        y = (data['avg_flows_loss'][i] / data['avg_flows_rate'][i]) * 100
        x = data['flows'][i]
        alg_data.append([alg, x, y])

data = pd.DataFrame(alg_data, columns=['alg', 'flows', 'avg_flows_loss_rate'])
data = data.sort_values(by=['alg', 'flows'], ignore_index=True)
print(data)

fig, ax = plt.subplots()

for key, grp in data.groupby(['alg']):
    ax = grp.plot(ax=ax, kind='line', x='flows', y='avg_flows_loss_rate', label=key)

plt.legend(loc='best')
plt.xlabel('Number of flows')
plt.ylabel('Flow loss rate (%)')
plt.show()
