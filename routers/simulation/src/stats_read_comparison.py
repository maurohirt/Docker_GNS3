from helper import get_results, visualization, graph_population
import numpy as np
from config import n_nodes, parent_folder, alg, n_flows_max, n_flows_min, n_flow_step, n_tries
import json

DB_DIR = 'dbs/20_nodes_net_SW/'
BASE_OUT_DIR = 'output/20_nodes_net_SW/'

algs = ['OSPF', 'Threshold', 'Weighted']

for alg in [x.lower() for x in algs]:
    flow_sequence = range(n_flows_min, n_flows_max, n_flow_step)

    avg_flows_rate = []
    avg_flows_loss = []
    tot_flows_rate = []
    tot_flows_loss = []

    for n_flow in flow_sequence:
        it_avg_flows_rate = []
        it_avg_flows_loss = []
        it_tot_flows_rate = []
        it_tot_flows_loss = []

        for n_try in range(n_tries):
            fname = f'{DB_DIR}{alg}/{n_nodes}_nodes_{n_try:02d}_{n_flow:04d}_db'
            print(fname)
            results = get_results(fname)
            it_avg_flows_rate.append(results.avg_flows_rate)
            it_avg_flows_loss.append(results.avg_flows_loss)
            it_tot_flows_rate.append(results.tot_flows_rate)
            it_tot_flows_loss.append(results.tot_flows_loss)

        avg_flows_rate.append(np.mean(it_avg_flows_rate))
        avg_flows_loss.append(np.mean(it_avg_flows_loss))
        tot_flows_rate.append(np.mean(it_tot_flows_rate))
        tot_flows_loss.append(np.mean(it_tot_flows_loss))

    stats = {
        'alg': alg,
        'flows': list(flow_sequence),
        'avg_flows_rate': avg_flows_rate,
        'avg_flows_loss': avg_flows_loss,
        'tot_flows_rate': tot_flows_rate,
        'tot_flows_loss': tot_flows_loss
    }

    print(stats)

    open(f'{BASE_OUT_DIR}{alg}.json', 'w').write(json.dumps(stats))
