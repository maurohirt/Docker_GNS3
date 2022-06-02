#!/usr/bin/env python3

n_nodes = 20
n_flows_max = 650
n_flows_min = 50
n_flow_step = 50
n_tries = 5

parent_folder = "dbs"
alg='ospf' # ospf, weighted, or threshold (requires [threshold, mult_fact])
alg_params=[[0.7],[10]]
