from helper import get_results, visualization, graph_population
import numpy as np
from config import n_nodes, parent_folder, alg
import json


folder = str(n_nodes)+"_nodes_net"
i_init = 1

BASE_OUT_FILE = 'output/{0}_{1}_{2}.json'

j_set = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 15, 30]
for j in j_set:
    results = []
    i_end = j
    for i in range(i_init, i_end+1):
        db_name = f"{n_nodes}_nodes_{i:02d}_db"
        db_pos = "./"+parent_folder+"/"+folder+"/"+db_name
        results.append(get_results(db_pos))
        G = graph_population("./"+parent_folder+"/"+folder, db_name)
        #vis = visualization(G)
        # vis.view_utilization(db_pos+".html")
        # vis.view_link_type(db_pos+"_link_type.html")
    avg_flows_rate = []
    avg_flows_loss = []
    tot_flows_rate = []
    tot_flows_loss = []
    for result in results:
        avg_flows_rate.append(result.avg_flows_rate)
        avg_flows_loss.append(result.avg_flows_loss)
        tot_flows_rate.append(result.tot_flows_rate)
        tot_flows_loss.append(result.tot_flows_loss)

    stats = {
        'mean': {
            'tot_flows_loss': np.mean(tot_flows_loss),
            'tot_flows_rate': np.mean(tot_flows_rate),
            'tot_flows_loss_rate': np.mean(np.divide(tot_flows_loss, tot_flows_rate))
        },
        'std': {
            'tot_flows_loss': np.std(tot_flows_loss),
            'tot_flows_rate': np.std(tot_flows_rate),
            'tot_flows_loss_rate': np.std(np.divide(tot_flows_loss, tot_flows_rate))
        },
        'min': {
            'tot_flows_loss': min(tot_flows_loss),
            'tot_flows_rate': min(tot_flows_rate),
            'tot_flows_loss_rate': min(np.divide(tot_flows_loss, tot_flows_rate))
        },
        'max': {
            'tot_flows_loss': max(tot_flows_loss),
            'tot_flows_rate': max(tot_flows_rate),
            'tot_flows_loss_rate': max(np.divide(tot_flows_loss, tot_flows_rate))
        }
    }
    print(j, stats)
    open(BASE_OUT_FILE.format(alg, n_nodes, j), 'w').write(json.dumps(stats))
