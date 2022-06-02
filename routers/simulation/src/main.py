##########################################################################
# Copyright (C) 2022 HARMONIA Project
#
# Main simulation caller for HARMONIA's Tiny SRv6 Controller (Not ready)
#
# Annotations on simulation:
# - remove DB functions
# - Reuse functions flows_effect (and its auxiliar functions), and calculate_path
# - Test using random_dest_flow_gen/single_des_flow_gen
#
##########################################################################

from helper import simulation, visualization
import networkx as nx
import random
import time
from config import n_nodes, parent_folder, alg_params, n_flows_max, n_flows_min, n_flow_step, n_tries


pce_node = 2
max_int = 8
max_links_out = 6

algs = ['ospf', 'threshold', 'weighted']

if __name__ == "__main__":
    for flow in range(n_flows_min, n_flows_max, n_flow_step):
        for alg in algs:
            random.seed(42)
            print(f"{flow} flows")
            for i in range(n_tries):
                print(i+1, "/", n_tries)
                s = simulation()
                db_folder = f"{n_nodes}_nodes_net_SW/{alg}/"
                db_name = f"{n_nodes}_nodes_{i:02d}_{flow:04d}_db"
                s.make_db(parent_folder, db_folder, db_name)

                while True:
                    seed = random.random()

                    domain_name = "A"
                    s.db_init()
                    s.small_world_topo_gen(
                        domain_name, n_nodes, pce_node, 4, 0.1)
                    # s.db_population_with_names(
                    #   domain_name, n_nodes, pce_node, max_int, max_links_out, seed=seed)

                    if nx.is_strongly_connected(s.G):
                        break

                try:
                    print("try")
                    s.flow_gen("A", "A", flow, mode="rate_set_prob", rate_set=[5, 10, 50],
                               rate_prob_dstr=[0.4, 0.3, 0.3], seed=seed)
                except Exception as e:
                    print("ERROR", e)

                # s.change_link_capacity(mode="cust_cap_set",cap_set=[1,2])
                s.change_link_capacity(mode="prob_cap_set", cap_set=[
                    40, 150, 1000, 10000], prob_cap_set=[0.2, 0.3, 0.2, 0.3])

                s.flows_effect(alg, alg_params)

                s.flow_delay_calc()

            print(s.domains)
            print(s.G)
